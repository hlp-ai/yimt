import pysbd
from sentence_splitter import split_text_into_sentences
from indicnlp.tokenize.sentence_tokenize import sentence_split



def split_sentences(text, lang="en"):
    """Segment paragraph into sentences

    Args:
        text: paragraph string
        lang: language code string

    Returns:
        list of sentences
    """
    languages_splitter = ["ca", "cs", "da", "de", "el", "en", "es", "fi", "fr", "hu", "is", "it",
                          "lt", "lv", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr"]
    languages_indic = ["as", "bn", "gu", "hi", "kK", "kn", "ml", "mr", "ne", "or", "pa", "sa",
                       "sd", "si", "ta", "te"]
    languages_pysbd = ["en", "hi", "mr", "zh", "es", "am", "ar", "hy", "bg", "ur", "ru", "pl",
                       "fa", "nl", "da", "fr", "my", "el", "it", "ja", "de", "kk", "sk"]

    languages = languages_splitter + languages_indic + languages_pysbd
    lang = lang if lang in languages else "en"

    text = text.strip()

    if lang in languages_pysbd:
        segmenter = pysbd.Segmenter(language=lang, clean=True)
        sentences = segmenter.segment(text)
    elif lang in languages_splitter:
        sentences = split_text_into_sentences(text, lang)
    elif lang in languages_indic:
        sentences = sentence_split(text, lang)

    return sentences


def paragraph_tokenizer(text, lang="en"):
    """Replace sentences with their indexes, and store indexes of newlines
    Args:
        text (str): Text to be indexed

    Returns:
        sentences (list): List of sentences
        breaks (list): List of indexes of sentences and newlines
    """
    text = text.strip()
    paragraphs = text.splitlines(True)

    breaks = []
    sentences = []

    for paragraph in paragraphs:
        if paragraph == "\n":
            breaks.append("\n")
        else:
            paragraph_sentences = split_sentences(paragraph, lang)

            breaks.extend(
                list(range(len(sentences), +len(sentences) + len(paragraph_sentences)))
            )
            breaks.append("\n")
            sentences.extend(paragraph_sentences)

    # Remove the last newline
    breaks = breaks[:-1]

    return sentences, breaks


def paragraph_detokenizer(sentences, breaks):
    """Restore original paragraph format from indexes of sentences and newlines

    Args:
        sentences (list): List of sentences
        breaks (list): List of indexes of sentences and newlines

    Returns:
        text (str): Text with original format
    """
    output = []

    for br in breaks:
        if br == "\n":
            output.append("\n")
        else:
            output.append(sentences[br] + " ")

    text = "".join(output)
    return text


def may_combine_paragraph(text):
    paragraphs = text.split("\n")
    txt = ""
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i].strip()
        i += 1
        if len(p) == 0:  # blank paragraph
            txt = txt + "\r\n\r\n"
        else:
            txt = txt + " " + p

    return txt
