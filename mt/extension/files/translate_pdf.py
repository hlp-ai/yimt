"""PDF file translation"""
import argparse
import logging
import os
import re
import fitz

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLine, LTChar
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame, KeepInFrame

from service.mt import translator_factory
from service.utils import detect_lang


logging.getLogger("pdfminer").setLevel(logging.WARNING)


fonts = {
    'zh': 'SimSun',
    'en': 'Arial',
    # Add more languages and their corresponding fonts here
}

pdfmetrics.registerFont(TTFont('SimSun', os.path.join(os.path.dirname(__file__), 'SimSun.ttf')))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(fontName='SimSun', name='Song', fontSize=9, wordWrap='CJK'))

p_chars_lang_independent = re.compile(r"[0123456789+\-*/=~!@$%^()\[\]{}<>\|,\.\?\"]")
p_en_chars = re.compile(r"[a-zA-Z]+")

font_size_list = []
dimlimit = 0  # 100  # each image side must be greater than this
relsize = 0  # 0.05  # image : image size ratio must be larger than this (5%)
abssize = 0  # 2048  # absolute image size limit 2 KB: ignore if smaller
imgdir = "output"  # found images are stored in this subfolder

if not os.path.exists(imgdir):  # make subfolder if necessary
    os.mkdir(imgdir)


def recoverpix(doc, item):
    xref = item[0]  # xref of PDF image
    smask = item[1]  # xref of its /SMask

    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        if pix0.n > 3:
            ext = "pam"
        else:
            ext = "png"

        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {  # create dictionary expected by caller
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png"),
        }
    return doc.extract_image(xref)


def extract_draw_save(source_pdf_fn, target_pdf_fn=None):
    if target_pdf_fn is None:
        paths = os.path.splitext(source_pdf_fn)
        translated_fn = paths[0] + "-translated" + paths[1]
    else:
        translated_fn = target_pdf_fn

    in_pdf = fitz.open(source_pdf_fn)
    page_count = in_pdf.page_count
    out_pdf = fitz.open(translated_fn)
    for i in range(page_count):
        page = in_pdf[i]
        paths = page.get_drawings()  # extract existing drawings

        if i >= len(out_pdf):  # 如果 outpdf 中没有第 i 页，那么添加一个新的页面
            out_pdf.new_page(width=page.rect.width, height=page.rect.height)

        outpage = out_pdf[i]
        shape = outpage.new_shape()  # make a drawing canvas for the output page
        # define some output page with the same dimensions

        # loop through the paths and draw them
        for path in paths:
            # draw each entry of the 'items' list
            for item in path["items"]:  # these are the draw commands
                if item[0] == "l":  # line
                    shape.draw_line(item[1], item[2])
                elif item[0] == "re":  # rectangle
                    shape.draw_rect(item[1])
                elif item[0] == "qu":  # quad
                    shape.draw_quad(item[1])
                elif item[0] == "c":  # curve
                    shape.draw_bezier(item[1], item[2], item[3], item[4])
                else:
                    raise ValueError("unhandled drawing", item)

            keys = ["fill", "color", "dashes", "even_odd", "closePath", "lineJoin", "lineCap", "width",
                    "stroke_opacity", "fill_opacity"]
            ensure_values(path, keys)
            lineCap = path.get("lineCap", [0])
            if isinstance(lineCap, int):
                lineCap = [lineCap]
            shape.finish(
                fill=path["fill"],  # fill color
                color=path["color"],  # line color
                dashes=path["dashes"],  # line dashing
                even_odd=path["even_odd"],  # control color of overlaps
                closePath=path["closePath"],  # whether to connect last and first point
                lineJoin=path["lineJoin"],  # how line joins should look like
                lineCap=max(lineCap),  # how line ends should look like
                width=path["width"],  # line width
                stroke_opacity=path["stroke_opacity"],
                fill_opacity=path["fill_opacity"],
            )

        shape.commit()

    return out_pdf


def ensure_values(dictionary, keys, default_value=1):
    for key in keys:
        if key not in dictionary or dictionary[key] is None:
            dictionary[key] = default_value


def extract_img_save(source_pdf_fn, target_pdf_fn):
    if target_pdf_fn is None:
        paths = os.path.splitext(source_pdf_fn)
        translated_fn = paths[0] + "-translated" + paths[1]
    else:
        translated_fn = target_pdf_fn

    in_pdf = fitz.open(source_pdf_fn)
    page_count = in_pdf.page_count  # number of pages
    out_pdf = fitz.open(translated_fn)

    xreflist = []
    imglist = []
    for pno in range(page_count):
        page = in_pdf[pno]
        il = in_pdf.get_page_images(pno)
        # print(il)
        imglist.extend([x[0] for x in il])
        for img in il:
            xref = img[0]
            img_rect = page.get_image_rects(xref)
            # print(img_rect)
            if xref in xreflist:
                continue
            width = img[2]
            height = img[3]
            if min(width, height) <= dimlimit:
                continue
            image = recoverpix(in_pdf, img)
            n = image["colorspace"]
            imgdata = image["image"]

            if len(imgdata) <= abssize:
                continue
            if len(imgdata) / (width * height * n) <= relsize:
                continue

            imgfile = os.path.join(imgdir, "img%05i.%s" % (xref, image["ext"]))
            print(imgfile)
            fout = open(imgfile, "wb")
            fout.write(imgdata)
            fout.close()

            xreflist.append(xref)
            out_pdf[pno].insert_image(rect=img_rect[0], stream=imgdata)

    imglist = list(set(imglist))
    # print(len(set(imglist)), "images in total")
    # print(imglist)
    # print(len(xreflist), "images extracted")
    # print(xreflist)
    return out_pdf


def get_fontsize(block):
    for line in block:
        if isinstance(line, LTTextLine):
            for char in line:
                if isinstance(char, LTChar):
                    return char.size


def remove_lang_independent(t):
    return re.sub(p_chars_lang_independent, "", t)


def has_en_char(t):
    return re.search(p_en_chars, t) is not None


def should_translate_en(txt):
    if not has_en_char(txt):
        return False

    tokens = txt.split()

    def is_translatable(token):
        if len(token) == 1:
            return False
        token = remove_lang_independent(token)
        if len(token) <= 1:
            return False

        return True

    return any(list(map(is_translatable, tokens)))


def preprocess_txt(t):
    return t.replace("-\n", "").replace("\n", " ").replace("<", "&lt;").strip()


def print_to_canvas(t, x, y, w, h, pdf, ft, tgt_lang="zh"):
    h = max(24, h)
    w = max(24, w)
    frame = Frame(x, y, w, h, showBoundary=0)

    font = fonts.get(tgt_lang, 'SimSun')  # 根据目标语言获取字体，如果没有对应的字体，则使用 'SimSun' 作为默认字体
    ft = round(ft)  # 将字体大小四舍五入到最接近的整数
    style_name = font + str(ft)
    if style_name not in styles:  # 如果样式不存在，则添加新样式
        styles.add(ParagraphStyle(fontName=font, name=style_name, fontSize=ft, wordWrap='CJK'))  # 使用获取到的字体
    story = [Paragraph(t, styles[style_name])]
    story_inframe = KeepInFrame(w, h, story)
    frame.addFromList([story_inframe], pdf)


def get_pdf_page_count(filename):
    with open(filename, 'rb') as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)
        return resolve1(document.catalog['Pages'])['Count']


def is_translatable(txt, source_lang):
    if source_lang == "en":
        return should_translate_en(txt)

    return True


def translate_pdf_auto(pdf_fn, source_lang="auto", target_lang="zh", translation_file=None, callbacker=None):
    if translation_file is None:
        paths = os.path.splitext(pdf_fn)
        translated_fn = paths[0] + "-translated" + paths[1]
    else:
        translated_fn = translation_file

    translator = None

    # MIN_WIDTH = 7
    # MIN_HEIGHT = 7

    total_pages = get_pdf_page_count(pdf_fn)

    pdf = canvas.Canvas(translated_fn)
    p = 1

    for page_layout in extract_pages(pdf_fn):  # for each page in pdf file
        print("*"*20, "Page", p, "*"*20, "\n")
        to_translate_blocks = []
        to_translate_texts = []
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                x, y, w, h = int(element.x0), int(element.y0), int(element.width), int(element.height)
                t = element.get_text()
                ft = get_fontsize(element)
                t = preprocess_txt(t)
                block = (x, y, w, h, t)

                # if w < MIN_WIDTH or h < MIN_HEIGHT:
                #     print("***TooSmall", block)
                #     print_to_canvas(t, x, y, w, h, pdf, ft, target_lang)
                #     continue

                if not is_translatable(t, source_lang):
                    print("***未翻译", block)
                    print_to_canvas(t, x, y, w, h, pdf, ft, target_lang)
                    continue

                print("***翻译", block)
                to_translate_blocks.append(block)
                to_translate_texts.append(t)

        if translator is None:
            if source_lang == "auto":
                source_lang = detect_lang(t)

            translator = translator_factory.get_translator(source_lang, target_lang)

            if translator is None:
                raise ValueError("给定语言不支持: {}".format(source_lang + "-" + target_lang))

            if callbacker:
                callbacker.set_tag(pdf_fn)

        translations = translator.translate_list(to_translate_texts, source_lang, target_lang)
        for i in range(len(to_translate_blocks)):
            x, y, w, h, t = to_translate_blocks[i]
            print_to_canvas(translations[i], x, y, w, h, pdf, ft, target_lang)

        if callbacker:
            callbacker.report(total_pages, p, fid=pdf_fn)

        pdf.showPage()
        p += 1

    pdf.save()

    # 保存绘制内容到翻译文档
    tf_draw = extract_draw_save(pdf_fn, translated_fn)
    tf_draw.saveIncr()

    # 保存图片到翻译文档
    translated_file = extract_img_save(pdf_fn, tf_draw)
    translated_file.saveIncr()

    return translated_file.name


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("PDF File Translator")
    arg_parser.add_argument("--to_lang", type=str, default="zh", help="target language")
    arg_parser.add_argument("--input_file", type=str, required=True, help="file to be translated")
    arg_parser.add_argument("--output_file", type=str, default=None, help="translation file")
    args = arg_parser.parse_args()

    to_lang = args.to_lang
    in_file = args.input_file
    out_file = args.output_file

    callback = None

    translated_fn = translate_pdf_auto(in_file, target_lang=to_lang, translation_file=out_file, callbacker=callback)

    import webbrowser

    webbrowser.open(in_file)
    webbrowser.open(translated_fn)
