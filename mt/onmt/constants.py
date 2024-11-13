"""Define constant values used across the project."""


class DefaultTokens(object):
    PAD = "<blank>"
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"
    MASK = "<mask>"
    VOCAB_PAD = "averyunlikelytoken"
    SENT_FULL_STOPS = [".", "?", "!"]
    PHRASE_TABLE_SEPARATOR = "|||"
    ALIGNMENT_SEPARATOR = " ||| "
    SEP = "｟newline｠"
    MASK_BEFORE = "｟_mask_before_｠"


class CorpusName(object):
    VALID = "valid"
    TRAIN = "train"
    INFER = "infer"


class CorpusTask(object):
    TRAIN = "train"
    VALID = "valid"
    INFER = "infer"


class ModelTask(object):
    LANGUAGE_MODEL = "lm"
    SEQ2SEQ = "seq2seq"
