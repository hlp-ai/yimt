import os

import sentencepiece as spm


def load_spm(sp_model_path):
    """Load SentencePiece model from file"""
    return spm.SentencePieceProcessor(model_file=sp_model_path)


def get_file_name(p):
    return os.path.basename(p)


def get_sp_prefix(corpus_path, vocab_size):
    corpus_path = get_file_name(corpus_path)
    return "{}-sp-{}".format(corpus_path, vocab_size)


def get_tok_file(corpus_path):
    corpus_path = get_file_name(corpus_path)
    return corpus_path + ".tok"