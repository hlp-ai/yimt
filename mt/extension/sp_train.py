"""Train SentencePiece model from corpus"""
import argparse
import sentencepiece as spm


def train_spm(corpus_fn,
              model_prefix,
              vocab_size,
              model_type="bpe",
              coverage=0.9999,
              num_sentences=5000000,
              normalization_rule_name="nmt_nfkc",
              remove_extra_whitespaces=True,
              add_dummy_prefix=False,
              split_digits=False,
              user_defined_symbols_file=None):
    """Train a SentencePiece model"""
    if user_defined_symbols_file is not None:
        user_defined_symbols = []
        with open(user_defined_symbols_file) as uf:
            for line in uf:
                line = line.strip()
                if len(line) > 0:
                    user_defined_symbols.append(line)

        spm.SentencePieceTrainer.train(input=corpus_fn,
                                       model_prefix=model_prefix,
                                       vocab_size=vocab_size,
                                       model_type=model_type,
                                       character_coverage=coverage,
                                       input_sentence_size=num_sentences,
                                       shuffle_input_sentence=True,
                                       normalization_rule_name=normalization_rule_name,
                                       remove_extra_whitespaces=remove_extra_whitespaces,
                                       add_dummy_prefix=add_dummy_prefix,
                                       split_digits=split_digits,
                                       user_defined_symbols=user_defined_symbols)
    else:
        spm.SentencePieceTrainer.train(input=corpus_fn,
                                   model_prefix=model_prefix,
                                   vocab_size=vocab_size,
                                   model_type=model_type,
                                   character_coverage=coverage,
                                   input_sentence_size=num_sentences,
                                   shuffle_input_sentence=True,
                                   normalization_rule_name=normalization_rule_name,
                                   remove_extra_whitespaces=remove_extra_whitespaces,
                                       split_digits=split_digits,
                                   add_dummy_prefix=add_dummy_prefix)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--corpus", required=True, help="Corpus file path")
    argparser.add_argument("--sp_prefix", default=None, help="SentencePiece model path prefix")
    argparser.add_argument("--vocab_size", type=int, default=32000, help="Vocab size")
    argparser.add_argument("--max_sentences", type=int, default=5000000, help="Max number of sentences for training")
    argparser.add_argument("--coverage", type=float, default=0.9999, help="Vocab coverage")
    argparser.add_argument("--normalization", default="nmt_nfkc", help="normalization_rule_name:nmt_nfkc/identity")
    argparser.add_argument("--remove_sp", type=bool, default=True, help="remove_extra_whitespaces")
    argparser.add_argument("--user_sym_file", type=str, default=None, help="user_defined_symbols_file")
    args = argparser.parse_args()

    if args.sp_prefix is None:
        sp_prefix = "{}-sp-{}".format(args.corpus, args.vocab_size)
    else:
        sp_prefix = args.sp_prefix

    train_spm(args.corpus, sp_prefix, args.vocab_size,
              coverage=args.coverage, num_sentences=args.max_sentences,
              normalization_rule_name=args.normalization,
              remove_extra_whitespaces=args.remove_sp,
              user_defined_symbols_file=args.user_sym_file)