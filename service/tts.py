import os

from vits.api import TTS
from vits.infer_zh import TTS_ZH

langcode_map = {
    "vi": "vie",
    "en": "eng",
    "zh": "zho"
}


class AudioGenerators:

    def __init__(self, models_dir=r"D:\kidden\github\yimt\pretrained\tts\vits"):
        self.generators = {}
        self.models_dir = models_dir

    def generate(self, txt, lang):
        if lang not in self.supported_languages():
            return None

        if lang in self.generators:
            generator = self.generators[lang]
        else:
            print("Loading TTS for {}...".format(lang))
            if lang == "zho":
                generator = TTS_ZH(config_path=os.path.join(self.models_dir, lang, "config.json"),
                 model_path=os.path.join(self.models_dir, lang, "G_100000.pth"))
            else:
                generator = TTS(lang, models_dir=self.models_dir)

            self.generators[lang] = generator

        output = generator.synthesize(txt)

        return [{"audio": output[0],
                 "sr": output[1]}]

    @staticmethod
    def supported_languages():
        return ["eng", "zho", "vie"]

    @staticmethod
    def to3letter(lang_code2letter):
        return langcode_map[lang_code2letter]


if __name__ == '__main__':
    g = AudioGenerators()
    r = g.generate("这只是个简单的运行测试。", lang="zho")
    print(r)
