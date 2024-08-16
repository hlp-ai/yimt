# 将SentencePiece产生的词典转换成训练使用的词典
import argparse
import math

from onmt.constants import DefaultTokens

OMIT = (DefaultTokens.UNK, DefaultTokens.BOS, DefaultTokens.EOS)


def convert(lines):
    for line in lines:
        w, c = line.rstrip('\n').split("\t")
        if w in OMIT:
            continue
        c = math.exp(float(c)) * 1000000
        c = int(c) + 1
        yield w, c


if __name__ == '__main__':
    parser = argparse.ArgumentParser("转换SentencePiece词典")
    parser.add_argument("-i", "--input", type=str, required=True, help="SentencePiece词典路径")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出词典路径")

    args = parser.parse_args()

    sp_vocab_fn = args.input  # SP词典路径
    vocab_fn = args.output  # 输出词典路径
    with open(sp_vocab_fn, encoding="utf-8") as f, open(vocab_fn, "w", encoding="utf-8") as out:
        for c, w in convert(f):
            out.write('{}\t{}\n'.format(c, w))
