"""txt纯文本文件翻译"""
import os
import argparse

from extension.files.utils import TranslationProgress
from service.mt import translator_factory
from service.utils import detect_lang


def translate_txt_auto(txt_fn, source_lang="auto", target_lang="zh", translation_file=None, callbacker=None):
    if translation_file is None:
        paths = os.path.splitext(txt_fn)
        translated_txt_fn = paths[0] + "-translated" + paths[1]
    else:
        translated_txt_fn = translation_file

    if callbacker:
        callbacker.set_info("读取源文档...", txt_fn)

    txt = open(txt_fn, encoding="utf-8").read()  # TODO: 大文本文件一次读入有问题

    if source_lang == "auto":
        source_lang = detect_lang(txt)

    translator = translator_factory.get_translator(source_lang, target_lang)

    if translator is None:
        raise ValueError("给定语言不支持: {}".format(source_lang+"-"+target_lang))

    translation = translator.translate_paragraph(txt, source_lang, target_lang, callbacker, txt_fn)

    if callbacker:
        callbacker.set_info("翻译完成，写出翻译结果", txt_fn)

    out_f = open(translated_txt_fn, "w", encoding="utf-8")
    out_f.write(translation)
    out_f.close()

    return translated_txt_fn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("纯文本文档翻译")
    arg_parser.add_argument("-tl", "--to_lang", type=str, default="zh", help="目标语言")
    arg_parser.add_argument("-i", "--input", type=str, required=True, help="待翻译文件")
    arg_parser.add_argument("-o", "--output", type=str, default=None, help="译文文件")
    args = arg_parser.parse_args()

    in_file = args.input
    out_file = args.output
    to_lang = args.to_lang

    callback = TranslationProgress()

    translated_txt_fn = translate_txt_auto(in_file, target_lang=to_lang, translation_file=out_file, callbacker=callback)

    import webbrowser
    webbrowser.open(in_file)
    webbrowser.open(translated_txt_fn)
