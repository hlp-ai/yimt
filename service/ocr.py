import numpy

from easyocr.api import OCR
from easyocr.config import all_lang_list


class TextRecognizers:

    def __init__(self, models_dir='D:\kidden\github\yimt\pretrained\ocr\easyocr'):
        self.models_dir = models_dir  # 模型根目录
        self._recognizers = {}  # 语言到识别器字典

    def recognize(self, img, lang):
        """ 对给定语言的图片进行OCR

        参数:
          img: 图片文件路径
          lang: 图片语言

        返回:
          None: 语言不支持
          文本框识别列表，每个文本框包括4坐标位置、文本和分数
        """
        if lang not in self.supported_languages():
            return None

        if lang in self._recognizers:
            recognizer = self._recognizers[lang]
        else:
            print("Loading OCR for {}...".format(lang))
            recognizer = OCR(lang, self.models_dir)
            self._recognizers[lang] = recognizer

        output = recognizer.recognize(img)
        result = []
        for pos, text, score in output:
            result.append({"pos": numpy.array(pos).tolist(),
                           "text": text,
                           "score": score})

        return result

    @staticmethod
    def supported_languages():
        """支持的语言列表"""
        return all_lang_list


if __name__ == "__main__":
    import sys
    import json

    lang = sys.argv[1]
    image_path = sys.argv[2]
    recognizers = TextRecognizers()

    result = recognizers.recognize(image_path, lang)
    print(result)
    print(json.dumps(result))