import os
import time
from time import strftime
import re


class TMRecord:

    def __init__(self, lang_pair=None, source=None, target=None):
        self.lang_pair = lang_pair
        self.source = source
        self.target = target

        if self.lang_pair:
            self.sl, self.tl = self.lang_pair.split("-")

    def __str__(self):
        return BasicTMSaver.RECORD_PATTERN.format(self.lang_pair, self.source, self.target)

    def __hash__(self):
        return hash(self.lang_pair + self.source + self.target)

    def __eq__(self, other):
        return self.lang_pair==other.lang_pair and self.source==other.source and self.target==other.target


class TMLoader:

    def tm_list(self):
        pass


class BasicTMLoader(TMLoader):

    pattern = re.compile("<lang-pair>([^<]+?)</lang-pair>\n<source>([^<]+?)</source>\n<target>([^<]+?)</target>",
                         re.DOTALL)

    def __init__(self, tm_file):
        self.tm_records = []
        with open(tm_file, encoding="utf-8") as f:
            text = f.read()

            matches = self.pattern.findall(text)
            for match in matches:
                self.tm_records.append(TMRecord(match[0].strip(),
                                                match[1].strip(),
                                                match[2].strip()))

    def tm_list(self):
        return self.tm_records


class TMSaver:

    def save(self, tm):
        pass

    def save_info(self, lang_pair, source, target):
        pass

    def close(self):
        pass

    def flush(self):
        pass


class BasicTMSaver(TMSaver):
    RECORD_PATTERN = "<lang-pair>\n{}\n</lang-pair>\n<source>\n{}\n</source>\n<target>\n{}\n</target>\n\n"

    def __init__(self, tm_dir=None, tm_file=None, overwrite=False):
        if tm_dir is None:
            tm_dir = os.path.join(os.path.dirname(__file__), "tm")
        if not os.path.exists(tm_dir):
            os.makedirs(tm_dir, exist_ok=True)
        self.fn_prefix = os.path.join(tm_dir, strftime("%Y%m%d.tm", time.localtime()) if tm_file is None else tm_file)
        if overwrite:
            self.tm_f = open(self.fn_prefix, "w", encoding="utf-8")
        else:
            self.tm_f = open(self.fn_prefix, "a", encoding="utf-8")

    def save(self, tm):
        record = self.RECORD_PATTERN.format(tm.lang_pair, tm.source, tm.target)
        self.tm_f.write(record)

    def save_info(self, lang_pair, source, target):
        if isinstance(source, list):
            assert len(source) == len(target)
            for i in range(len(source)):
                record = self.RECORD_PATTERN.format(lang_pair, source[i], target[i])
                self.tm_f.write(record)
        else:
            record = self.RECORD_PATTERN.format(lang_pair, source, target)
            self.tm_f.write(record)

    def close(self):
        self.tm_f.close()

    def flush(self):
        self.tm_f.flush()


tm_saver = None


def get_tm_saver(tm_dir=None, tm_file=None):
    global tm_saver
    if tm_saver is not None:
        return tm_saver

    tm_saver = BasicTMSaver(tm_dir=tm_dir, tm_file=tm_file)
    return tm_saver


if __name__ == "__main__":
    tm = TMRecord()
    tm.lang_pair = "en-zh"
    tm.source = "This is a book."
    tm.target = "这是一本书。"

    saver = get_tm_saver()
    saver.save(tm)
    saver.save(tm)
    saver.close()

    tm_loader = BasicTMLoader("./tm/20240821.tm")
    unique_tm = set()
    print(len(tm_loader.tm_list()))
    for r in tm_loader.tm_list():
        print(r)
        unique_tm.add(r)

    print(len(unique_tm))
