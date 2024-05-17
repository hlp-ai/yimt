import threading

from onmt.inference_engine import InferenceEnginePY
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts
from service.sentence_splitter import paragraph_tokenizer, paragraph_detokenizer


class Translator:

    def __init__(self, conf_file, lang_pair):
        base_args = ["-config", conf_file]
        parser = self._get_parser()
        self.opt = parser.parse_args(base_args)
        ArgumentParser._get_all_transform_translate(self.opt)
        self.opt.share_vocab = False

        self.engine = InferenceEnginePY(self.opt)
        self.lang_pair = lang_pair
        self.from_lang, self.to_lang = self.lang_pair.split("-")

    def translate_list(self, texts):
        scores, preds = self.engine.infer_list(texts)

        return [p[0] for p in preds]

    def translate_paragraph(self, texts):
        """Translate text paragraphs

        the text will be segmented into paragraphs, and then paragraph segmented into sentences.
        the format of text will be remained.

        Args:
            texts: text to be translated

        Returns:
             translated text with paragraphs
        """
        source_sents, breaks = paragraph_tokenizer(texts, self.from_lang)

        translations = self.translate_list(source_sents)

        translation = paragraph_detokenizer(translations, breaks)

        return translation

    def directions(self):
        return [self.lang_pair]

    def _get_parser(self):
        parser = ArgumentParser()
        opts.translate_opts(parser)
        return parser


class Translators(object):

    def __init__(self):
        self.translators= {}

    def get_translator(self, source_lang, target_lang, debug=False):
        lang_pair = source_lang + "-" + target_lang
        return self.translators[lang_pair]


translator_factory = Translators()


if __name__ == "__main__":
    conf_file = "D:/kidden/github/yimt/mt/toy-enzh/infer.yaml"
    translator = Translator(conf_file, "en-zh")

    texts = ["how are you?", "i am a teacher."]
    preds = translator.translate_list(texts)
    print(preds)