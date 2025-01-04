import os

import fasttext

fasttext_model = fasttext.load_model(os.path.join(os.path.dirname(__file__), "lid.176.ftz"))


def detect_lang(text, k=1, lib="fasttext"):
    """Detect language of text

    :param text: text to be detected
    :param k: number of languages to return
    :param lib: which detection lib used
    :return:
    """
    if lib == "fasttext":
        text = text[:200].lower() if len(text) > 200 else text.lower()
        prediction = fasttext_model.predict(text.replace("\n", " "), k=k)
        lang1 = prediction[0][0][9:]
        if k > 1:
            lang2 = prediction[0][1][9:]
            return lang1, lang2
        else:
            return lang1
    elif lib == "pycld2":
        import pycld2 as cld2
        isReliable, textBytesFound, details, vectors = cld2.detect(text, returnVectors=True)
        langs = [d[1] for d in details]
        # langs = list(filter(lambda e: e!="un", langs))
        if k > 1:
            return langs[:k]
        else:
            return langs[0]
    elif lib == "langid":
        import langid

        r = langid.classify(text)
        if k > 1:
            return [e[0] for e in r][:k]
        else:
            return r[0][0]
    else:
        raise ValueError


if __name__ == "__main__":
    enzh1 = "this is a very interesting book. 这本书很有趣。"
    print(detect_lang(enzh1, 2))

    print(detect_lang("неправильный формат идентификатора дн назад", 2))
