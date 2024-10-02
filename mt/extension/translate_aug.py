import os
import re
import sys

from extension.utils import pair_to_single, single_to_pair


def aug(in_path, conf_file):
    print("翻译扩展" + in_path)
    x_file = in_path + ".src"
    en_file = in_path + ".en"
    pair_to_single(in_path, x_file, en_file)
    out_zh_path = en_file + ".tozh"
    x_zh_tsv = in_path + ".aug2zh"

    if os.path.exists(x_zh_tsv):
        print("扩展文件已存在")
        return

    cmd = "python -m onmt.bin.translate -config {} -src {} -output {}".format(conf_file, en_file, out_zh_path)

    cf = os.popen(cmd)
    lines = cf.readlines()
    for line in lines:
        print(line.strip())

    single_to_pair(x_file, out_zh_path, x_zh_tsv)


def aug_dir(in_dir, conf_file):
    fils = os.listdir(in_dir)
    files = [os.path.join(in_dir, f) for f in fils]

    in_files = []
    for f in files:
        if re.match(r"\d+$", f):
            in_files.append(f)
    print(in_files)

    if len(in_files) == 0:
        return

    for f in in_files:
        aug(f, conf_file)


if __name__ == "__main__":
    conf_path = sys.argv[1]
    in_path = sys.argv[2]

    if os.path.isdir(in_path):
        aug_dir(in_path, conf_path)
    else:
        aug(in_path, conf_path)
