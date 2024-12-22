import argparse
import os

from bs4 import BeautifulSoup, Comment, Doctype

from extension.files.utils import should_translate, TranslationProgress
from service.mt import translator_factory
from service.utils import detect_lang


def collect_tag(markup_str, no_translatable_tags=['style', 'script', 'head', 'meta', 'link'], lang="en"):
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
            if not should_translate(t, lang):  # 过滤不需要翻译的
                continue

            to_translated_elements.append(element)
            to_translated_txt.append(element.string)

    return markup, to_translated_elements, to_translated_txt


def translate_tag_list(markup_strs, source_lang="auto", target_lang="zh", callbacker=None):
    markups = []
    to_translated_tags = []
    # to_translated_strs = []
    to_translated_list = []
    for s in markup_strs:
        m, es, ts = collect_tag(s, lang=source_lang)
        markups.append(m)
        to_translated_tags.append(es)
        # to_translated_strs.append(ts)
        for s in ts:
            to_translated_list.append(s)

    if source_lang == "auto":
        source_lang = detect_lang(" ".join(to_translated_list))

    from service.mt import translator_factory
    translator = translator_factory.get_translator(source_lang, target_lang)
    if translator is None:
        raise ValueError("给定语言对不支持: {}".format(source_lang + "-" + target_lang))

    translations = translator.translate_list(to_translated_list, sl=source_lang, tl=target_lang, callbacker=callbacker)

    idx = 0
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

    if callbacker:
        callbacker.set_info("读取源文档...", in_fn)

    in_fn_lower = in_fn.lower()
    html_txt = open(in_fn, encoding="utf-8").read()
    if in_fn_lower.endswith(".html") or in_fn_lower.endswith(".xhtml") or in_fn_lower.endswith(".htm"):
        soup = BeautifulSoup(html_txt, "html.parser")
    elif in_fn_lower.endswith(".xml") or in_fn_lower.endswith(".sgml"):
        soup = BeautifulSoup(html_txt, "xml")

    body = soup
    to_translated_elements = []
    to_translated_txt = []
    for element in body.findAll(text=True):  # 对所有文本节点
        if not element.parent.name in ['style', 'script', 'head', 'meta', 'link']:
            if type(element.string) == Comment:  # 不翻译注释
                continue

            if type(element.string) == Doctype: # 不翻译文档类型节点
                continue

            t = element.string
            if not should_translate(t):  # 过滤掉不需要翻译的
                continue

            to_translated_elements.append(element)
            to_translated_txt.append(element.string)

    if source_lang == "auto":
        source_lang = detect_lang(t)

    translator = translator_factory.get_translator(source_lang, target_lang)

    if translator is None:
        raise ValueError("给定语言对不支持: {}".format(source_lang+"-"+target_lang))

    translations = translator.translate_list(to_translated_txt, sl=source_lang, tl=target_lang,
                                             callbacker=callbacker,
                                             fn=in_fn)

    for e, t in zip(to_translated_elements, translations):
        e.replaceWith(t)

    if callbacker:
        callbacker.set_info("翻译完成，写出翻译结果", in_fn)

    out_f = open(translated_fn, "w", encoding="utf-8")
    out_f.write(soup.prettify())
    out_f.close()

    return translated_fn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("标记语言文件翻译")
    arg_parser.add_argument("-tl", "--to_lang", type=str, default="zh", help="目标语言")
    arg_parser.add_argument("-i", "--input", type=str, required=True, help="待翻译文件")
    arg_parser.add_argument("-o", "--output", type=str, default=None, help="译文文件")
    args = arg_parser.parse_args()

    in_file = args.input
    out_file = args.output
    to_lang = args.to_lang

    callback = TranslationProgress()

    translated_fn = translate_ml_auto(in_file, target_lang=to_lang, translation_file=out_file,
                                      callbacker=callback)

    import webbrowser

    webbrowser.open(in_file)
    webbrowser.open(translated_fn)
