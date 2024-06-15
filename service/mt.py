import os

import yaml

from onmt.inference_engine import InferenceEnginePY
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts
from service.split_text import paragraph_tokenizer, paragraph_detokenizer
from service.tm import get_tm_saver


class Progress:

    def __init__(self):
        self._tag = ""

    def report(self, total, done):
        print(self._tag, total, done)

    def set_tag(self, tag):
        self._tag = tag


class Translator:

    def __init__(self, conf_params, batch_size=32):
        self.conf_file = conf_params["model_or_config_dir"]
        self.lang_pairs = conf_params["directions"]

        self._init(self.conf_file, batch_size);
        # base_args = ["-config", conf_file]
        # parser = self._get_parser()
        # self.opt = parser.parse_args(base_args)
        # ArgumentParser._get_all_transform_translate(self.opt)
        # self.opt.share_vocab = False
        #
        # self.engine = InferenceEnginePY(self.opt)
        # self.lang_pairs = lang_pairs
        #
        # self.batch_size = batch_size  # 句子数
        #
        # self.tm_saver = get_tm_saver()

    def _init(self, conf_file, batch_size=32):
        base_args = ["-config", conf_file]
        parser = self._get_parser()
        self.opt = parser.parse_args(base_args)
        ArgumentParser._get_all_transform_translate(self.opt)
        self.opt.share_vocab = False

        print("加载翻译引擎: {}".format(self.conf_file))

        self.engine = InferenceEnginePY(self.opt)
        self.batch_size = batch_size  # 句子数
        self.tm_saver = get_tm_saver()

    def translate_list(self, texts, sl, tl, callbacker=None):
        results = []
        total = len(texts)
        done = 0
        for i in range(0, len(texts), self.batch_size):
            if i + self.batch_size < len(texts):
                to_translate = texts[i:i + self.batch_size]
            else:
                to_translate = texts[i:]

            to_translate = self._preprocess(to_translate, sl, tl)

            scores, preds = self.engine.infer_list(to_translate)
            translations = [p[0] for p in preds]

            lang_pair = sl + "-" + tl

            self.tm_saver.save_info(lang_pair, to_translate, translations)

            results.extend(translations)
            done += len(to_translate)

            if callbacker:
                callbacker.report(total, done)

        return results

    def _preprocess(self, texts, sl, tl):
        return texts;

    def translate_paragraph(self, texts, sl, tl, callbacker=None):
        """Translate text paragraphs

        the text will be segmented into paragraphs, and then paragraph segmented into sentences.
        the format of text will be remained.

        Args:
            texts: text to be translated

        Returns:
             translated text with paragraphs
        """
        source_sents, breaks = paragraph_tokenizer(texts, sl)

        translations = self.translate_list(source_sents, sl, tl, callbacker)

        translation = paragraph_detokenizer(translations, breaks)

        return translation

    def support(self, lang_pair):
        for p in self.lang_pairs:
            if p == lang_pair:
                return True

        return False

    def supported(self):
        return self.lang_pairs

    def _get_parser(self):
        parser = ArgumentParser()
        opts.translate_opts(parser)
        return parser


class ZhEnJaArTranslator(Translator):

    # def __init__(self, conf_file):
    #     super(ZhEnJaArTranslator, self).__init__(conf_file, lang_pairs = ["zh-ar", "zh-en", "zh-en"])

    # def translate_list(self, texts, sl=None, tl=None, callbacker=None):
    #     if tl == "ar":
    #         prefix = "<toar>"
    #     elif tl == "ja":
    #         prefix = "<toja>"
    #     else:
    #         prefix = "<toen>"
    #
    #     texts = [prefix + t for t in texts]
    #     scores, preds = self.engine.infer_list(texts)
    #
    #     return [p[0] for p in preds]

    def _preprocess(self, texts, sl, tl):
        if tl == "ar":
            prefix = "<toar>"
        elif tl == "ja":
            prefix = "<toja>"
        else:
            prefix = "<toen>"

        texts = [prefix + t for t in texts]
        return texts



class Translators(object):

    def __init__(self, config_path=os.path.join(os.path.dirname(__file__), "translators.yml")):
        self.config_file = config_path

        self.translators, self.lang_pairs, self.languages = self.load_config()

        # self.languages = [
        #     {"code": "zh", "name": "Chinese", "cname": "中文"},
        #     {"code": "en", "name": "English", "cname": "英文"},
        #     {"code": "ja", "name": "Japanese", "cname": "日文"}
        # ]
        # self.lang_pairs = ["zh-en", "zh-ja"]
        self.from_langs = ["zh"]
        self.to_langs = ["en", "ja"]

        # translator = ZhEnJaArTranslator("./infer.yaml")
        #
        # self.translators = {"zh-en": translator,
        #                     "zh-ja": translator}

    def load_config(self):
        translators = {}
        with open(self.config_file, encoding="utf-8") as config_f:
            config = yaml.safe_load(config_f.read())

        lang_pairs = set()
        for name, params in config.get("translators").items():
            translators[name] = ZhEnJaArTranslator(params)
            for p in params["directions"]:
                lang_pairs.add(p)

        lang_infos = []
        for lang in config.get("languages"):
            lang_infos.append(lang)

        return translators, lang_pairs, lang_infos

    def support_languages(self):
        return self.lang_pairs, self.from_langs, self.to_langs, self.languages

    def get_translator(self, source_lang, target_lang, debug=False):
        lang_pair = source_lang + "-" + target_lang

        if lang_pair not in self.lang_pairs:
            return None

        for name, translator in self.translators.items():
            if translator.support(lang_pair):
                return translator


translator_factory = Translators()


if __name__ == "__main__":
    # conf_file = "D:/kidden/github/yimt/mt/toy-enzh/infer.yaml"
    # translator = Translator(conf_file, "en-zh")
    #
    # texts = ["how are you?", "i am a teacher."]
    # preds = translator.translate_list(texts)
    # print(preds)

    # conf_file = "./infer.yaml"
    # translator = ZhEnJaArTranslator(conf_file)

    translator = translator_factory.get_translator("zh", "en")

    texts = ["你在做什么？", "我是一名教师。"]
    preds = translator.translate_list(texts, sl="zh", tl="en")
    print(preds)

    preds = translator.translate_list(texts, sl="zh", tl="ja")
    print(preds)