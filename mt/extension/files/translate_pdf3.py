"""PDF file translation"""
import argparse
import os
import pymupdf

from extension.files.pdf.copy_drawings import copy_drawings
from extension.files.pdf.copy_image import copy_images
from extension.files.pdf.translate_pdf_text import get_candidate_block, font_dict
from service.mt import translator_factory



def translate_pdf_auto(pdf_fn, source_lang="auto", target_lang="zh", translation_file=None, callbacker=None):
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
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
        copy_drawings(page, outpage)
        copy_images(page, outpage, doc)

        blocks = page.get_text("dict")["blocks"]
        candidates = []
        for block in blocks:
            cb = get_candidate_block(block)
            candidates.extend(cb)

        if translator is None:
            translator = translator_factory.get_translator(source_lang, target_lang)

            if translator is None:
                raise ValueError("给定语言不支持: {}".format(source_lang + "-" + target_lang))

        shape = outpage.new_shape()

        for c in candidates:
            text = c["text"]
            text = text.replace("-\n", "").replace("\n", " ").replace("<", "&lt;").strip()
            if len(text) == 0:
                continue
            toks = text.split()
            avg_len = sum([len(t) for t in toks]) / len(toks)
            if avg_len > 3 and len(toks) > 1:
                c["text"] = translator.translate_paragraph(text, source_lang, target_lang)
                c["font"] = "china-ss"

            print(c)
            shape.insert_textbox(c["bbox"], c["text"],
                                 fontsize=c["size"] * 0.75,
                                 fontname=c["font"],
                                 rotate=0, lineheight=c["size"]*0.2)
            # shape.draw_rect(pymupdf.Rect(c["bbox"]))

        shape.finish()
        shape.commit()

        if callbacker:
            callbacker.report(total_pages, i+1, fid=pdf_fn)

    outpdf.save(translated_fn)

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
