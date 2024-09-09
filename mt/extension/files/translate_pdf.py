"""PDF file translation"""
import argparse
import io
import logging
import os
import re
import tempfile
import random

import pymupdf
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Frame, KeepInFrame

from extension.files.pdf.copy_drawings import copy_drawings
from extension.files.pdf.copy_image import copy_images
from service.mt import translator_factory
from service.split_text import paragraph_tokenizer
from service.utils import detect_lang


# 关闭pdfminer的部分日志
logging.getLogger("pdfminer").setLevel(logging.WARNING)


# 各语言文本使用字体
fonts = {
    'zh': 'SimSun',
    'en': 'Arial',
}


temp_pdf_dir = tempfile.mkdtemp()


def get_temp_pdf():
    rand_id = random.randint(0, 10000)
    return os.path.join(temp_pdf_dir, "{}.pdf".format(rand_id))


pdfmetrics.registerFont(TTFont('SimSun', os.path.join(os.path.dirname(__file__), 'SimSun.ttf')))
styles = getSampleStyleSheet()

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


# font_dict = {
#     "en": "helv",
#     "zh": "china-ss",
# }


def span_len(span):
    return len(span["text"].split())


# TODO: pymupdf的分段有问题，这里需要分成段落
def merge_block(block):
    sizes = []  # 各文本段字体大小
    for line in block["lines"]:
        for span in line["spans"]:
            sizes.append(span["size"])

    text = ""  # 文本块文本
    for line in block["lines"]:
        line_text = ""  # 行文本
        for span in line["spans"]:
            line_text += span["text"]
        line_text = line_text.strip()
        if line_text[-1] == "-":
            text += line_text[:-1]
        else:
            text += line_text + " "

    return [
        {"text": text.strip(),
         "bbox": block["bbox"],
         # "font": font_dict["en"],
         "size": min(sizes),
         "dir": (1.0, 0.0)}
    ]


def get_candidate_block(block):
    if len(block["lines"]) == 1:  # 单行文本块
        line = block["lines"][0]
        if len(line["spans"]) == 1:  # 单行单段
            span = line["spans"][0]
            return [
                {"text": span["text"],
                     "bbox": span["bbox"],
                     # "font": font_dict["en"],   # XXX
                     "size": span["size"],
                     "dir": line["dir"]}
            ]
        else:  # 单行多段
            lens = [span_len(s) for s in line["spans"]]
            if sum(lens)/len(line["spans"]) < 3:  # 每段很短，不合并
                return [
                    {"text": s["text"],
                     "bbox": s["bbox"],
                     # "font": font_dict["en"],
                     "size": s["size"],
                     "dir": (1.0, 0.0)} for s in line["spans"]
                ]
            else:  # 合并多个段
                return merge_block(block)
    else:  # 多行文本块
        fonts = []
        sizes = []
        lens = []
        for line in block["lines"]:
            for span in line["spans"]:
                fonts.append(span["font"])
                sizes.append(span["size"])
                lens.append(span_len(span))

        if sum(lens) / len(lens) < 3:  # 每段很短，不合并
            result = []
            for line in block["lines"]:
                result.extend([
                    {"text": s["text"],
                     "bbox": s["bbox"],
                     # "font": font_dict["en"],
                     "size": s["size"],
                     "dir": (1.0, 0.0)} for s in line["spans"]
                ])
            return result
        else:  # 合并多个段
            return merge_block(block)


def block_heigth(b):
    return b[3] - b[1]


def block_width(b):
    return b[2] - b[0]


def in_paragraph(line1, line2):
    eps = 3.0
    bbox1 = line1["bbox"]
    bbox2 = line2["bbox"]

    h1 = block_heigth(bbox1)
    w1 = block_width(bbox1)
    x11 = bbox1[0]
    x12 = bbox1[2]

    h2 = block_heigth(bbox2)
    w2 = block_width(bbox2)
    x21 = bbox2[0]
    x22 = bbox2[2]

    # if abs(h2-h1) < eps and abs(w2-w1) < eps and abs(x21-x11) < eps:
    #     return True

    # 行高度接近，且左边或右边对齐
    if abs(h2-h1) < eps and (abs(x21-x11) < eps or abs(x22-x12) <eps):
        return True

    return False


def merge_spans(line):
    text = ""
    for span in line["spans"]:
        text += span["text"]

    return text


def create_paragraph(block, start, end):
    x1, y1, x2, y2 = 999999.0, 999999.0,0.0, 0.0
    text = ""
    size = block["lines"][start]["spans"][0]["size"]
    for i in range(start, end+1):
        line = block["lines"][i]
        text_line = merge_spans(line)
        text += text_line + "\n"
        x1 = min(line["bbox"][0], x1)
        y1 = min(line["bbox"][1], y1)
        x2 = max(line["bbox"][2], x2)
        y2 = max(line["bbox"][3], y2)

    return {"text": text.strip(),
            "bbox": (x1, y1, x2, y2),
            "size": size}


def get_paragraphs(block):
    n_lines = len(block["lines"])
    if n_lines == 1:  # 单行文本块
        line = block["lines"][0]
        bbox = line["bbox"]

        text = ""
        size = line["spans"][0]["size"]
        text = merge_spans(line)

        return [{"text": text.strip(),
                 "bbox": bbox,
                 "size": size}]
    else:  # 多行文本块
        paragraphs = []
        i = 0
        start_idx = i
        while i < n_lines:
            if i+1 < n_lines:
                if not in_paragraph(block["lines"][i], block["lines"][i+1]):
                    # 创建start_idx到i的段落
                    p = create_paragraph(block, start_idx, i)
                    paragraphs.append(p)
                    start_idx = i + 1

            i += 1

        if start_idx <= i:
            p = create_paragraph(block, start_idx, i-1)
            paragraphs.append(p)

        return paragraphs


def print_to_page(block, canvas, page_h, tgt_lang="zh"):
    t = block["text"].replace("<", "&lt;")  # reportlab需要转义
    x1, y1, x2, y2 = block["bbox"]
    h = y2 - y1
    w = x2 - x1

    # MuPDF和PDF规范坐标系不同，y坐标变换，x坐标不需要
    y1 = page_h - y1
    y2 = page_h - y2

    x = x1
    y = y2 - 12  # XXX: 人为减12，变换问题？

    font_size = block["size"]
    # XXX: 人为设置最小高度和宽度
    h = max(24, h)
    w = max(48, w)
    frame = Frame(x, y, w, h, showBoundary=0)

    # 根据目标语言获取字体，如果没有对应的字体，则使用 'SimSun' 作为默认字体
    font = fonts.get(tgt_lang, 'SimSun')
    font_size = round(font_size)  # 将字体大小四舍五入到最接近的整数

    # 创建字体样式
    style_name = font + str(font_size)
    if style_name not in styles:  # 如果样式不存在，则添加新样式
        styles.add(ParagraphStyle(fontName=font, name=style_name, fontSize=font_size, wordWrap='CJK'))  # 使用获取到的字体
    story = [Paragraph(t, styles[style_name])]
    story_inframe = KeepInFrame(w, h, story)  # XXX: 译文容易大小缩减
    frame.addFromList([story_inframe], canvas)


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

    print("复制图形和图像，提取文本信息...")
    if callbacker:
        callbacker.set_info("读取源文档...", pdf_fn)

    text_pages = []  # 每页文本块
    for page in doc:
        # 生成输出页面
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)

        # 复制形状
        copy_drawings(page, outpage)

        # 复制图形
        copy_images(page, outpage, doc)

        # 页面文本块
        blocks = page.get_text("dict")["blocks"]
        candidates = []
        for block in blocks:
            if block["type"] != 0:  # XXX:为什么这里有图片？
                continue

            # # 获得候选翻译文本块
            # cb = get_candidate_block(block)
            # candidates.extend(cb)

            paragraphs = get_paragraphs(block)
            candidates.extend(paragraphs)

        text_pages.append(candidates)

    temp_pdf = get_temp_pdf()  # 中间临时pdf文件
    outpdf.save(temp_pdf)

    print("翻译和输出...")

    translator = None
    input_df = PdfReader(open(temp_pdf, "rb"))
    output_pdf = open(translated_fn, "wb")
    output = PdfWriter()
    for i, template_page in enumerate(input_df.pages):  # 循环每一页
        print("页面{}".format(i))

        packet = io.BytesIO()  # 画布写出目的地
        canvas_draw = Canvas(packet,
                             pagesize=(input_df.pages[i].mediabox.width, input_df.pages[i].mediabox.height))

        page_h = float(input_df.pages[i].mediabox.height)  # 页面高度，用户坐标转换

        blocks = text_pages[i]
        for block in blocks:
            print(block)
            text = block["text"]
            text = text.replace("-\n", "").replace("\n", " ").strip()
            if len(text) == 0:
                continue

            if translator is None:
                if source_lang == "auto":
                    source_lang = detect_lang(text)  # TODO: 语言检测更安全些

                translator = translator_factory.get_translator(source_lang, target_lang)
                if translator is None:
                    raise ValueError("给定语言对不支持: {}".format(source_lang + "-" + target_lang))

            toks = text.split()
            avg_len = sum([len(t) for t in toks]) / len(toks)
            if avg_len > 3 and len(toks) > 1:
                source_sents, breaks = paragraph_tokenizer(text, source_lang)
                # source_sents = text.splitlines(True)
                source_sents_tag = []
                to_translate = []
                for s in source_sents:
                    if should_translate_en(s):
                        source_sents_tag.append((True, s))
                        to_translate.append(s)
                    else:
                        source_sents_tag.append((False, s))

                translations = translator.translate_list(to_translate, source_lang, target_lang, callbacker)
                result = ""
                j = 0
                for i in range(len(source_sents_tag)):
                    if source_sents_tag[i][0]:
                        result += translations[j] + " "
                        j += 1
                    else:
                        result += source_sents_tag[i][1] + " "

                block["text"] = result.strip()

            # print(text)
            print(block)

            # 写出到画布
            print_to_page(block, canvas_draw, page_h)

        canvas_draw.save()

        # 合并文本页面
        template_page.merge_page(PdfReader(packet).pages[0])
        output.add_page(template_page)

        output.write(output_pdf)

        if callbacker:
            callbacker.report(total_pages, i+1, fid=pdf_fn)

    if callbacker:
        callbacker.set_info("翻译完成，写出翻译结果", pdf_fn)

    output_pdf.close()

    return translated_fn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("PDF文件翻译")
    arg_parser.add_argument("-tl", "--to_lang", type=str, default="zh", help="目标语言")
    arg_parser.add_argument("-i", "--input_file", type=str, required=True, help="输入PDF文件路径")
    arg_parser.add_argument("-o", "--output_file", type=str, default=None, help="翻译结果PDF文件路径")
    args = arg_parser.parse_args()

    to_lang = args.to_lang
    in_file = args.input_file
    out_file = args.output_file

    callback = None

    translated_fn = translate_pdf_auto(in_file, source_lang="en", target_lang=to_lang,
                                       translation_file=out_file, callbacker=callback)

    import webbrowser

    webbrowser.open(in_file)
    webbrowser.open(translated_fn)
