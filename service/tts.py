from vits.api import TTS
from vits.infer_zh import TTS_ZH


class AudioGenerators:

    def __init__(self):
        self.generators = {}

    def generate(self, txt, lang):
        if lang in self.generators:
            generator = self.generators[lang]
        else:
            if lang == "zh":
                generator = TTS_ZH(config_path=r"D:\kidden\github\yimt\pretrained\tts\zho\config.json",
                 model_path=r"D:\kidden\github\yimt\pretrained\tts\zho\G_100000.pth")
            else:
                generator = TTS(lang)

            self.generators[lang] = generator

        output = generator.synthesize(txt)

        return [{"audio": output[0],
                 "sr": output[1]}]

    @staticmethod
    def supported_languages():
        return ["en", "zh"]


if __name__ == '__main__':
    g = AudioGenerators()
    r = g.generate("这只是个简单的运行测试。", lang="zh")
    print(r)
