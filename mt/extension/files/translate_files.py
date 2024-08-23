"""Doc translation"""
import argparse
import os

from extension.files.translate_docx import translate_docx_auto
from extension.files.translate_html import translate_ml_auto
from extension.files.translate_pdf import translate_pdf_auto
from extension.files.translate_ppt import translate_ppt_auto
from extension.files.translate_txt import translate_txt_auto


def support(file_type):
    return file_type in [".txt", ".pdf", ".html", ".htm", ".xhtml", ".xml", ".sgml", ".docx", ".pptx"]


def get_type(fn):
    return os.path.splitext(fn)[1]


# 标签文档类型
TAG_FILES = [".html", ".htm", ".xhtml", ".xml", ".sgml"]


def translate_doc(src_fn, source_lang="auto", target_lang="zh", translation_file=None, callbacker=None):
    file_type = get_type(src_fn)

    if file_type == ".txt":
        return translate_txt_auto(src_fn, source_lang, target_lang, translation_file, callbacker=callbacker)
    elif file_type == ".pdf":
        return translate_pdf_auto(src_fn, source_lang, target_lang, translation_file, callbacker=callbacker)
    elif file_type == ".docx":
        return translate_docx_auto(src_fn, source_lang, target_lang, translation_file, callbacker=callbacker)
    elif file_type in TAG_FILES:
        return translate_ml_auto(src_fn, source_lang, target_lang, translation_file, callbacker=callbacker)
    elif file_type == ".pptx":
        return translate_ppt_auto(src_fn, source_lang, target_lang, translation_file, callbacker=callbacker)
    else:
        return None


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("File Translator")
    arg_parser.add_argument("--source_lang", type=str, default="auto", help="source language for file")
    arg_parser.add_argument("--to_lang", type=str, default="zh", help="target language")
    arg_parser.add_argument("--input_file", type=str, required=True, help="file to be translated")
    arg_parser.add_argument("--output_file", type=str, default=None, help="translation file")
    args = arg_parser.parse_args()

    source_lang = args.source_lang
    to_lang = args.to_lang
    in_file = args.input_file
    out_file = args.output_file

    translated_fn = translate_doc(in_file, source_lang, to_lang, out_file)

    import webbrowser
    webbrowser.open(translated_fn)
    webbrowser.open(in_file)
