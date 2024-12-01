"""PDF file translation"""
import argparse
import os
import re
from pprint import pprint

import pymupdf

from extension.files.pdf.copy_drawings import copy_drawings
from extension.files.pdf.copy_image import copy_images
from extension.files.pdf.utils import get_candidate_block, blocks_for_translation
from service.mt import translator_factory


debug = False


def translate_pdf_auto(pdf_fn, source_lang="auto", target_lang="zh", translation_file=None, callbacker=None,
                       tm_saver=None):
    if translation_file is None:
        paths = os.path.splitext(pdf_fn)
        translated_fn = paths[0] + "-translated" + paths[1]
    else:
        translated_fn = translation_file

    translator = None

    doc = pymupdf.open(pdf_fn)
    total_pages = doc.page_count
    outpdf = pymupdf.open()

    for i, page in enumerate(doc):
        if debug:
            print("***Page{}***".format(i))

        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)

        # 拷贝绘制形状
        # copy_drawings(page, outpage)

        # 拷贝图形
        copy_images(page, outpage, doc)

        # blocks = page.get_text("dict")["blocks"]
        # candidates = []
        # for block in blocks:
        #     if block["type"] != 0:  # 忽略图形块，inline图形不包含在get_images返回中
        #         continue
        #
        #     if debug:
        #         pprint(block)
        #
        #     cb = get_candidate_block(block)
        #     candidates.extend(cb)
        candidates = blocks_for_translation(page)

        if translator is None:
            translator = translator_factory.get_translator(source_lang, target_lang)

            if translator is None:
                raise ValueError("给定语言不支持: {}".format(source_lang + "-" + target_lang))

        shape = outpage.new_shape()

        for c in candidates:
            print(c)
            text = c["text"]
            text = re.sub(r"\s{2,}", " ", text)
            # text = text.replace("-\n", "").replace("\n", " ").replace("<", "&lt;").strip()
            text = text.replace("- ", "").replace("<", "&lt;").strip()
            if len(text) == 0:
                continue
            toks = text.split()
            avg_len = sum([len(t) for t in toks]) / len(toks)
            if avg_len > 3 and len(toks) > 1:
                c["text"] = translator.translate_paragraph(text, source_lang, target_lang)

                if tm_saver:
                    tm_saver.save_info(source_lang+"-"+target_lang, text, c["text"])

            outpage.insert_htmlbox(c["bbox"], c["text"],
                                   css="* {text-align: justify;}")

            if debug:
                outpage.draw_rect(pymupdf.Rect(c["bbox"]),
                              color=(1, 0, 0),
                              dashes="[3] 0")

        copy_drawings(page, outpage)

        shape.finish()
        shape.commit()

        if callbacker:
            callbacker.report(total_pages, i+1, fid=pdf_fn)

    outpdf.save(translated_fn)

    return translated_fn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("PDF File Translator")
    arg_parser.add_argument("-tl", "--to_lang", type=str, default="zh", help="target language")
    arg_parser.add_argument("-i", "--input_file", type=str, required=True, help="file to be translated")
    arg_parser.add_argument("-o", "--output_file", type=str, default=None, help="translation file")
    arg_parser.add_argument("-d", "--debug", action="store_true", help="debug or not")
    args = arg_parser.parse_args()

    to_lang = args.to_lang
    in_file = args.input_file
    out_file = args.output_file

    debug = args.debug

    callback = None

    from service.tm import BasicTMSaver
    tm_saver = BasicTMSaver(tm_file=os.path.basename(in_file).replace(" ", "_")+".tm")

    print(tm_saver.fn_prefix)

    translated_fn = translate_pdf_auto(in_file, source_lang="en", target_lang=to_lang, translation_file=out_file,
                                       callbacker=callback, tm_saver=tm_saver)

    import webbrowser

    webbrowser.open(in_file)
    webbrowser.open(translated_fn)
