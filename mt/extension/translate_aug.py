import os
import sys

from extension.utils import pair_to_single, single_to_pair


def aug(in_path, conf_file):
    x_file = in_path + ".src"
    en_file = in_path + ".en"
    pair_to_single(in_path, x_file, en_file)
    out_zh_path = en_file + ".tozh"
    cmd = "python -m onmt.bin.translate -config {} -src {} -output {}".format(conf_file, en_file, out_zh_path)

    cf = os.popen(cmd)
    lines = cf.readlines()
    for line in lines:
        print(line.strip())

    x_zh_tsv = in_path + ".aug2zh"

    single_to_pair(x_file, out_zh_path, x_zh_tsv)


if __name__ == "__main__":
    conf_path = sys.argv[1]
    in_path = sys.argv[2]

    aug(in_path, conf_path)
