import threading
from langid import langid
from onmt.inference_engine import InferenceEnginePY
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts
from service.split_text import paragraph_tokenizer, paragraph_detokenizer


def detect_lang(text):
    """Detect language of text

    Args:
        text: text to be detected

    Returns:
        language code string
    """
    # text = remove_lang_independent(text)
    if all([c.isascii() for c in text]):
        return "en"
    return langid.classify(text)[0]


class Translator:

    def __init__(self, conf_file, lang_pair):
        base_args = ["-config", conf_file]
        parser = self._get_parser()
        self.opt = parser.parse_args(base_args)
        ArgumentParser._get_all_transform_translate(self.opt)
        self.opt.share_vocab = False

        self.engine = InferenceEnginePY(self.opt)
        self.lang_pair = lang_pair

    def translate_list(self, texts, sl=None, tl=None):
        scores, preds = self.engine.infer_list(texts)

        return [p[0] for p in preds]

    def translate_paragraph(self, texts, sl=None, tl=None):
        """Translate text paragraphs

        the text will be segmented into paragraphs, and then paragraph segmented into sentences.
        the format of text will be remained.

        Args:
            texts: text to be translated

        Returns:
             translated text with paragraphs
        """
        source_sents, breaks = paragraph_tokenizer(texts, sl)

        translations = self.translate_list(source_sents, sl, tl)

        translation = paragraph_detokenizer(translations, breaks)

        return translation

    def support(self, lang_pair):
        for p in self.directions():
            if p == lang_pair:
                return True

        return False

    def _get_parser(self):
        parser = ArgumentParser()
        opts.translate_opts(parser)
        return parser


class ZhEnJaArTranslator(Translator):

    def __init__(self, conf_file, lang_pair = ["zh-ar", "zh-en", "zh-en"]):
        super(ZhEnJaArTranslator, self).__init__(conf_file, lang_pair)

    def translate_list(self, texts, sl=None, tl=None):
        if tl == "ar":
            prefix = "<toar>"
        elif tl == "ja":
            prefix = "<toja>"
        else:
            prefix = "<toen>"

        texts = [prefix + t for t in texts]
        scores, preds = self.engine.infer_list(texts)

        return [p[0] for p in preds]


class Translators(object):

    def __init__(self):
        self.translators= {}

    def get_translator(self, source_lang, target_lang, debug=False):
        lang_pair = source_lang + "-" + target_lang
        return self.translators[lang_pair]


translator_factory = Translators()


if __name__ == "__main__":
    # conf_file = "D:/kidden/github/yimt/mt/toy-enzh/infer.yaml"
    # translator = Translator(conf_file, "en-zh")
    #
    # texts = ["how are you?", "i am a teacher."]
    # preds = translator.translate_list(texts)
    # print(preds)

    conf_file = "./infer.yaml"
    translator = ZhEnJaArTranslator(conf_file)

    texts = ["你在做什么？", "我是一名教师。"]
    preds = translator.translate_list(texts, tl="en")
    print(preds)

    preds = translator.translate_list(texts, tl="ja")
    print(preds)