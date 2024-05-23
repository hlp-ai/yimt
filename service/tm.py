import os
import time
from time import strftime


class TranslationRecord:

    def __init__(self):
        self.lang_pair = None
        self.original = None
        self.translation = None


class TMSaver:

    def save(self, tr):
        pass

    def save_info(self, lang_pair, original, translation):
        pass

    def close(self):
        pass

    def flush(self):
        pass


class BasicTMSaver(TMSaver):
    RECORD_PATTERN = "<lang-pair>\n{}\n</lang-pair>\n<original>\n{}\n</original>\n<translation>\n{}\n</translation>\n\n"

    def __init__(self, tm_dir=None):
        if tm_dir is None:
            tm_dir = "./tm"
        if not os.path.exists(tm_dir):
            os.makedirs(tm_dir, exist_ok=True)
        self.fn_prefix = os.path.join(tm_dir, strftime("%Y%m%d.tm", time.localtime()))
        self.tm_f = open(self.fn_prefix, "a", encoding="utf-8")

    def save(self, tr):
        record = self.RECORD_PATTERN.format(tr.lang_pair, tr.original, tr.translation)
        self.tm_f.write(record)

    def save_info(self, lang_pair, original, translation):
        if isinstance(original, list):
            assert len(original) == len(translation)
            for i in range(len(original)):
                record = self.RECORD_PATTERN.format(lang_pair, original[i], translation[i])
                self.tm_f.write(record)
        else:
            record = self.RECORD_PATTERN.format(lang_pair, original, translation)
            self.tm_f.write(record)

    def close(self):
        self.tm_f.close()

    def flush(self):
        self.tm_f.flush()


tm_saver = None


def get_tm_saver():
    global tm_saver
    if tm_saver is not None:
        return tm_saver

    tm_saver = BasicTMSaver()
    return tm_saver


if __name__ == "__main__":
    tm = TranslationRecord()
    tm.lang_pair = "en-zh"
    tm.original = "This is a book."
    tm.translation = "这是一本书。"

    saver = get_tm_saver()
    saver.save(tm)
    saver.save(tm)
    saver.close()
