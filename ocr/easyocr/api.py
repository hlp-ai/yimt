from easyocr.reader import Reader


class OCR:

    def __init__(self, lang, models_dir='D:\kidden\github\yimt\pretrained\ocr\easyocr'):
        if lang != "en":
            langs = [lang, 'en']
        else:
            langs = [lang]
        self._reader = Reader(langs, model_storage_directory=models_dir)

    def recognize(self, img_path):
        result = self._reader.readtext(img_path)
        return result


if __name__ == "__main__":
    ocr_fr = OCR("fr")
    r = ocr_fr.recognize('../examples/french.jpg')
    print(r)

    ocr_ja = OCR("ja")
    r = ocr_ja.recognize('../examples/japanese.jpg')
    print(r)