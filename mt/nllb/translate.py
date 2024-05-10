from torch.nn.utils.rnn import pad_sequence
import torch

from onmt.constants import DefaultTokens
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

src_prefix = "</s> eng_Latn"
tgt_prefix = "zho_Hans"

segmented_prefix = [src_prefix.split() + s for s in segmented]
print(segmented_prefix)

src_ids = [src_vocab(s) for s in segmented_prefix]
print(src_ids)

padidx = src_vocab[DefaultTokens.PAD]
src_ids = [torch.LongTensor(ids) for ids in src_ids]
src_ids_padded = pad_sequence(src_ids, batch_first=True,
                             padding_value=padidx)
print(src_ids_padded)

tgt = [[tgt_prefix]] * len(text)
print(tgt)
tgt_ids = [src_vocab(s) for s in tgt]
print(tgt_ids)
tgt_ids = torch.LongTensor(tgt_ids)

batch = {
    "src": src_ids_padded,
    "tgt": tgt_ids,
    "srclen": torch.LongTensor([len(s) for s in src_ids])
}

print(batch)

result = translator.translate_batch(batch, False)
print(result)
