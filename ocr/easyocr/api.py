import sys

from easyocr.reader import Reader


class OCR:

    def __init__(self, lang, models_dir='D:\kidden\github\yimt\pretrained\ocr\easyocr'):
        langs = [lang]
        self._reader = Reader(langs, model_storage_directory=models_dir)

    def recognize(self, img_path, paragraph=True):
        result = self._reader.readtext(img_path, paragraph=paragraph)
        return result


if __name__ == "__main__":
    lang = sys.argv[1]
    img_file = sys.argv[2]
    ocr = OCR(lang)
    r = ocr.recognize(img_file)
    print(r)
