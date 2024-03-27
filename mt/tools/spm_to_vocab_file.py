# converts a SentencePiece vocabulary to the format expected by dynamic data
# (essentially converts float expected counts to "fixed precision" int pseudo
# counts)
import sys
import math
from onmt.constants import DefaultTokens

OMIT = (DefaultTokens.UNK, DefaultTokens.BOS, DefaultTokens.EOS)


def convert(lines):
    for line in lines:
        w, c = line.rstrip('\n').split(None, 1)
        if w in OMIT:
            continue
        c = math.exp(float(c)) * 1000000
        c = int(c) + 1
        yield w, c


if __name__ == '__main__':
    sp_vocab_fn = sys.argv[1]
    vocab_fn = sys.argv[2]
    with open(sp_vocab_fn, encoding="utf-8") as f, open(vocab_fn, "w", encoding="utf-8") as out:
        for c, w in convert(f):
            out.write('{}\t{}\n'.format(c, w))
