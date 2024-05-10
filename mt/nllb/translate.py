from onmt.opts import config_opts, translate_opts
from onmt.translate.translator import build_translator
from onmt.utils.parse import ArgumentParser


parser = ArgumentParser()
config_opts(parser)
translate_opts(parser, dynamic=True)
argv = "-config D:/kidden/github/yimt/pretrained/mt/nllb/nllb-inference.yaml -src dummy-src"
opt = parser.parse_args(args=argv)
print(opt)

print("Loading model...")
translator = build_translator(opt)

src_vocab = translator.vocabs["src"]
print(len(src_vocab))
# for e in src_vocab.tokens_to_ids:
#     print(e)

import sentencepiece as spm
sp_model_src = spm.SentencePieceProcessor()
sp_model_src.Load(opt.src_subword_model)

text = ["how are you?", "what are you doing now?"]
segmented = sp_model_src.encode(text, out_type=str)
print(segmented)
