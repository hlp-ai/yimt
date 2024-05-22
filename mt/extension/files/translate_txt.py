"""TXT file translation"""
import os
import argparse

from service.mt import Progress, translator_factory
from service.utils import detect_lang


def translate_txt_auto(txt_fn, source_lang="auto", target_lang="zh", translation_file=None, callbacker=None):
    if translation_file is None:
        paths = os.path.splitext(txt_fn)
        translated_txt_fn = paths[0] + "-translated" + paths[1]
    else:
        translated_txt_fn = translation_file

    txt = open(txt_fn, encoding="utf-8").read()  # TODO: 大文本文件一次读入有问题

    if source_lang == "auto":
        source_lang = detect_lang(txt)

    translator = translator_factory.get_translator(source_lang, target_lang)

    if callbacker:
        callbacker.set_tag(txt_fn)

    translation = translator.translate_paragraph(txt, callbacker)

    out_f = open(translated_txt_fn, "w", encoding="utf-8")
    out_f.write(translation)
    out_f.close()

    return translated_txt_fn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("TXT File Translator")
    arg_parser.add_argument("--to_lang", type=str, default="zh", help="target language")
    arg_parser.add_argument("--input_file", type=str, required=True, help="file to be translated")
    arg_parser.add_argument("--output_file", type=str, default=None, help="translation file")
    args = arg_parser.parse_args()

    in_file = args.input_file
    out_file = args.output_file
    to_lang = args.to_lang

    callback = Progress()

    translated_txt_fn = translate_txt_auto(in_file, target_lang=to_lang, translation_file=out_file, callbacker=callback)

    import webbrowser
    webbrowser.open(in_file)
    webbrowser.open(translated_txt_fn)
