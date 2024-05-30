"""Tokenize file with SentencePiece"""
import argparse
import io

from extension.sp import load_spm


def tokenize_sp(sp_model, txt):
    """Tokenize text with SentencePiece

    :param sp_model: SentencePiece model
    :param txt: text or list of text
    :return: list of tokens
    """
    if not isinstance(txt, (list, tuple)):
        txt = [txt]
    tokens = sp_model.encode(txt, out_type=str)
    return tokens


def tokenize_file_sp(sp_model, in_fn, out_fn):
    """Tokenize file with SentencePiece model and output result into file"""
    if isinstance(sp_model, str):
        sp_model = load_spm(sp_model)
    in_f = io.open(in_fn, encoding="utf-8")
    out_f = io.open(out_fn, "w", encoding="utf-8")
    sentences = 0
    tokens = 0
    for s in in_f:
        s = s.strip()
        tok_s = tokenize_sp(sp_model, s)[0]
        sentences += 1
        tokens += len(tok_s)
        if len(tok_s) > 0:
            out_f.write(" ".join(tok_s) + "\n")
        else:
            out_f.write("\n")
        if sentences % 100000 == 0:
            print("Sentences:", sentences, "Tokens:", tokens)
    print("Sentences:", sentences, "Tokens:", tokens)
    in_f.close()
    out_f.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sp_model", required=True, help="SentencePiece model path")
    argparser.add_argument("--in_fn", required=True, help="Corpus file path")
    argparser.add_argument("--out_fn", default=None, help="Ouput file path")
    args = argparser.parse_args()

    if args.out_fn is None:
        out_fn = args.in_fn + ".tok"
    else:
        out_fn = args.out_fn

    sp = load_spm(args.sp_model)
    tokenize_file_sp(sp, args.in_fn, out_fn)
