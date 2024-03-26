import numpy

from easyocr.api import OCR
from easyocr.config import all_lang_list


class TextRecognizers:

    def __init__(self, models_dir='D:\kidden\github\yimt\pretrained\ocr\easyocr'):
        self.models_dir = models_dir

        self._recognizers = {}

    def recognize(self, img, lang):
        if lang not in self.supported_languages():
            return None

        if lang in self._recognizers:
            recognizer = self._recognizers[lang]
        else:
            recognizer = OCR(lang, self.models_dir)

        output = recognizer.recognize(img)
        result = []
        for pos, text, score in output:
            result.append({"pos": numpy.array(pos).tolist(),
                           "text": text,
                           "score": score})

        return result

    @staticmethod
    def supported_languages():
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