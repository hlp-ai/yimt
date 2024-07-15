"""PDF file translation"""
import argparse
import io
import logging
import os
import re
import pymupdf
from PyPDF2 import PdfReader, PdfWriter, Transformation

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Frame, KeepInFrame

from extension.files.pdf.copy_drawings import copy_drawings
from extension.files.pdf.copy_image import copy_images
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


font_dict = {
    "en": "helv",
    "zh": "china-ss",
}


def span_len(span):
    return len(span["text"].split())


def merge_block(block):
    sizes = []
    for line in block["lines"]:
        for span in line["spans"]:
            sizes.append(span["size"])

    text = ""
    for line in block["lines"]:
        line_text = ""
        for span in line["spans"]:
            line_text += span["text"]
        text += line_text + "\n"

    return [{"text":text, "bbox":block["bbox"], "font":font_dict["en"], "size":min(sizes), "dir":(1.0, 0.0)}]


def get_candidate_block(block):
    if len(block["lines"]) == 1:
        line = block["lines"][0]
        if len(line["spans"]) == 1:  # 单行单段
            span = line["spans"][0]
            return [{"text":span["text"], "bbox":span["bbox"], "font":font_dict["en"], "size":span["size"], "dir":line["dir"]}]
        else:  # 单行多段
            lens = [span_len(s) for s in line["spans"]]
            if sum(lens)/len(line["spans"]) < 3:  # 每段很短
                return [{"text":s["text"], "bbox":s["bbox"], "font":font_dict["en"], "size":s["size"], "dir":(1.0, 0.0)} for s in line["spans"]]
            else:
                return merge_block(block)
    else:  # 多行
        fonts = []
        sizes = []
        lens = []
        for line in block["lines"]:
            for span in line["spans"]:
                fonts.append(span["font"])
                sizes.append(span["size"])
                lens.append(span_len(span))

        if sum(lens) / len(lens) < 3:  # 每段很短
            result = []
            for line in block["lines"]:
                result.extend([{"text":s["text"], "bbox":s["bbox"], "font":font_dict["en"], "size":s["size"], "dir":(1.0, 0.0)} for s in line["spans"]])
            return result
        else:
            return merge_block(block)


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


def print_to_page(block, canvas, page_h, tgt_lang="zh"):
    t = block["text"]
    x1, y1, x2, y2 = block["bbox"]
    h = y2 - y1
    w = x2 - x1

    y1 = page_h - y1
    y2 = page_h - y2

    x = x1
    y = y2

    ft = block["size"]
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
    frame.addFromList([story_inframe], canvas)


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

    doc = pymupdf.open(pdf_fn)
    total_pages = doc.page_count
    outpdf = pymupdf.open()

    pages = []  # 每页文本块

    translator = None

    print("复制图形和图像，提取文本信息...")

    for page in doc:
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
        copy_drawings(page, outpage)
        copy_images(page, outpage, doc)

        blocks = page.get_text("dict")["blocks"]
        candidates = []
        for block in blocks:
            cb = get_candidate_block(block)
            candidates.extend(cb)
        pages.append(candidates)

    outpdf.save("temp.pdf")

    print("翻译和输出...")

    input_df = PdfReader(open("temp.pdf", "rb"))  # 输入
    output_pdf = open(translated_fn, "wb")  # 输出
    output = PdfWriter()
    for i, template_page in enumerate(input_df.pages):  # 循环每一页
        print("页面{}".format(i+1))
        packet = io.BytesIO()
        # 设置中文（如果不这样设置中文，中文会变成黑色的方块）
        # pdfmetrics.registerFont(TTFont("SimHei", "SimHei.ttf"))  # 步骤1
        canvas_draw = Canvas(packet,
                             pagesize=(input_df.pages[0].mediabox.width, input_df.pages[0].mediabox.height))

        page_h = float(input_df.pages[0].mediabox.height)

        if translator is None:
            translator = translator_factory.get_translator(source_lang, target_lang)

        blocks = pages[i]
        for c in blocks:
            text = c["text"]
            text = text.replace("-\n", "").replace("\n", " ").replace("<", "&lt;").strip()
            if len(text) == 0:
                continue

            toks = text.split()
            avg_len = sum([len(t) for t in toks]) / len(toks)
            if avg_len > 3 and len(toks) > 1:
                c["text"] = translator.translate_paragraph(text, source_lang, target_lang)

            print(c)

            print_to_page(c, canvas_draw, page_h)

        # canvas_draw.setFont("SimHei", 20)  # 支持中文
        # canvas_draw.drawString(100, 500, "随便添加一句话")  # 添加内容

        canvas_draw.save()

        template_page.add_transformation(Transformation().rotate(0).translate(tx=0, ty=0))
        template_page.merge_page(PdfReader(packet).pages[0])

        output.add_page(template_page)
        output.write(output_pdf)

    output_pdf.close()

    return translated_fn


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

    translated_fn = translate_pdf_auto(in_file, source_lang="en", target_lang=to_lang, translation_file=out_file, callbacker=callback)

    import webbrowser

    webbrowser.open(in_file)
    webbrowser.open(translated_fn)
