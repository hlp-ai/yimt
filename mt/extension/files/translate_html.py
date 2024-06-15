import argparse
import os

from bs4 import BeautifulSoup, Comment, Doctype


# def too_short(txt, lang):
#     if len(txt.strip()) == 0:
#         return True
#
#     if lang == "en" and len(txt.strip().split()) < 3:
#         return True
#
#     if lang == "zh" and len(list(jieba.cut(txt))) < 3:
#         return True
#
#     return False
from service.utils import detect_lang


def collect(markup_str, no_translatable_tags=['style', 'script', 'head', 'meta', 'link']):
    """收集标记中所有可以翻译的元素和文本"""
    markup = BeautifulSoup(markup_str, "html.parser")
    to_translated_elements = []
    to_translated_txt = []
    for element in markup.findAll(text=True):
        if not element.parent.name in no_translatable_tags:
            if type(element.string) == Comment:
                continue

            if type(element.string) == Doctype:
                continue

            t = element.string
            if len(t.strip()) == 0:
                continue

            to_translated_elements.append(element)
            to_translated_txt.append(element.string)

    return markup, to_translated_elements, to_translated_txt


def translate_tag_list(markup_strs, source_lang="auto", target_lang="zh", callbacker=None):
    markups = []
    to_translated_tags = []
    to_translated_strs = []
    to_translated_list = []
    for s in markup_strs:
        m, es, ts = collect(s)
        markups.append(m)
        to_translated_tags.append(es)
        to_translated_strs.append(ts)
        for s in ts:
            to_translated_list.append(s)

    if source_lang == "auto":
        source_lang = detect_lang(to_translated_list[0])

    from service.mt import translator_factory
    translator = translator_factory.get_translator(source_lang, target_lang)

    translations = translator.translate_list(to_translated_list, sl=source_lang, tl=target_lang, callbacker=callbacker)

    idx  = 0
    for i in range(len(markups)):
        replace(to_translated_tags[i], translations[idx:idx+len(to_translated_tags[i])])
        idx += len(to_translated_tags[i])

    return [m.prettify() for m in markups]


def replace(to_translated_elements, translations):
    """用翻译文本替换元素中原始文本"""
    for e, t in zip(to_translated_elements, translations):
        e.replaceWith(t)


def translate_ml_auto(in_fn, source_lang="auto", target_lang="zh", translation_file=None, callbacker=None):
    if translation_file is None:
        paths = os.path.splitext(in_fn)
        translated_fn = paths[0] + "-translated" + paths[1]
    else:
        translated_fn = translation_file

    in_fn = in_fn.lower()
    html_txt = open(in_fn, encoding="utf-8").read()
    if in_fn.endswith(".html") or in_fn.endswith(".xhtml") or in_fn.endswith(".htm"):
        soup = BeautifulSoup(html_txt, "html.parser")
    elif in_fn.endswith(".xml") or in_fn.endswith(".sgml"):
        soup = BeautifulSoup(html_txt, "xml")

    body = soup

    to_translated_elements = []
    to_translated_txt = []
    for element in body.findAll(text=True):
        if not element.parent.name in ['style', 'script', 'head', 'meta', 'link']:
            if type(element.string) == Comment:
                continue

            if type(element.string) == Doctype:
                continue

            t = element.string
            if len(t.strip()) == 0:
                continue

            to_translated_elements.append(element)
            to_translated_txt.append(element.string)

    if source_lang == "auto":
        source_lang = detect_lang(t)

    translator = translator_factory.get_translator(source_lang, target_lang)

    if callbacker:
        callbacker.set_tag(in_fn)

    translations = translator.translate_list(to_translated_txt, callbacker=callbacker)

    for e, t in zip(to_translated_elements, translations):
        e.replaceWith(t)

    out_f = open(translated_fn, "w", encoding="utf-8")
    out_f.write(soup.prettify())
    out_f.close()

    return translated_fn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("HTML/XML File Translator")
    arg_parser.add_argument("--to_lang", type=str, default="zh", help="target language")
    arg_parser.add_argument("--input_file", type=str, required=True, help="file to be translated")
    arg_parser.add_argument("--output_file", type=str, default=None, help="translation file")
    args = arg_parser.parse_args()

    to_lang = args.to_lang
    in_file = args.input_file
    out_file = args.output_file

    translated_fn = translate_ml_auto(in_file, target_lang=to_lang, translation_file=out_file)

    import webbrowser

    webbrowser.open(in_file)
    webbrowser.open(translated_fn)
