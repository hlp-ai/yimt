from onmt.inference_engine import InferenceEnginePY
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts


class Translator:

    def __init__(self, conf_file, lang_pair):
        base_args = ["-config", conf_file]
        parser = self._get_parser()
        self.opt = parser.parse_args(base_args)
        ArgumentParser._get_all_transform_translate(self.opt)
        self.opt.share_vocab = False

        self.engine = InferenceEnginePY(self.opt)
        self.lang_pair = lang_pair

    def translate_list(self, texts):
        scores, preds = self.engine.infer_list(texts)

        return [p[0] for p in preds]

    def directions(self):
        return [self.lang_pair]

    def _get_parser(self):
        parser = ArgumentParser()
        opts.translate_opts(parser)
        return parser


if __name__ == "__main__":
    conf_file = "D:/kidden/github/yimt/mt/toy-enzh/infer.yaml"
    translator = Translator(conf_file, "en-zh")

    texts = ["how are you?", "i am a teacher."]
    preds = translator.translate_list(texts)
    print(preds)