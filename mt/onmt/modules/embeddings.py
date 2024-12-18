""" Embeddings module """
import math

import torch
import torch.nn as nn
from torch.nn.utils import skip_init

from onmt.utils.logging import logger


class SequenceTooLongError(Exception):
    pass


# At the moment this class is only used by embeddings.Embeddings look-up tables
class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    emb is a 3d Tensor whose last dimension is the same length
    as the list.
    emb_out is the result of applying modules to emb elementwise.
    An optional merge parameter allows the emb_out to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        super(Elementwise, self).__init__(*args)

    def forward(self, emb):
        emb_ = [feat.squeeze(2) for feat in emb.split(1, dim=2)]
        emb_out = []

        # for some reason list comprehension is slower in this scenario
        for f, x in zip(self, emb_):
            emb_out.append(f(x))

        return emb_out[0]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dim (int): embedding size
    """

    def __init__(self, dim, enc_type, max_len=5000):
        if dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(dim)
            )
        if enc_type == "SinusoidalInterleaved":
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, dim, 2, dtype=torch.float)
                    * -(math.log(10000.0) / dim)
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
        elif enc_type == "SinusoidalConcat":
            half_dim = dim // 2
            pe = math.log(10000) / (half_dim - 1)
            pe = torch.exp(torch.arange(half_dim, dtype=torch.float) * -pe)
            pe = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * pe.unsqueeze(0)
            pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1).view(max_len, -1)
        else:
            raise ValueError("Choice of Position encoding is SinusoidalInterleaved or SinusoidalConcat.")

        pe = pe.unsqueeze(1)  # we keep pe (len x batch x dim) for back comp
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)  # 位置嵌入作为模块buffer
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        pe = self.pe.transpose(0, 1)  # (batch x len x dim)
        emb = emb * math.sqrt(self.dim)
        step = step or 0
        if pe.size(1) < step + emb.size(1):
            raise SequenceTooLongError(
                f"Sequence is {emb.size(1) + step} but PositionalEncoding is"
                f" limited to {self.pe.size(1)}. See max_len argument."
            )
        emb = emb + pe[:, step : emb.size(1) + step, :]

        return emb


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        word_padding_idx (int): padding index for words in the embeddings.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        dropout (float): dropout probability.
        sparse (bool): sparse embbedings default False
        freeze_word_vecs (bool): freeze weights of word vectors.
    """

    def __init__(
        self,
        word_vec_size,
        word_vocab_size,
        word_padding_idx,
        position_encoding=False,
        position_encoding_type="SinusoidalInterleaved",
        dropout=0,
        sparse=False,
        freeze_word_vecs=False,
    ):
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # The embedding matrix look-up tables. The first look-up table
        # is for words.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [
            skip_init(
                nn.Embedding,
                num_embeddings=vocab,
                embedding_dim=dim,
                padding_idx=pad,
                sparse=sparse,
            )
            for vocab, dim, pad in emb_params
        ]
        emb_luts = Elementwise(None, embeddings)

        self.embedding_size = word_vec_size

        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module("emb_luts", emb_luts)

        self.position_encoding = position_encoding
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

        if self.position_encoding:
            pe = PositionalEncoding(self.embedding_size, position_encoding_type)
            self.make_embedding.add_module("pe", pe)

        if freeze_word_vecs:
            self.word_lut.weight.requires_grad = False

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data.copy_(pretrained[:, : self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def forward(self, source, step=None):
        """Computes the embeddings for words.

        Args:
            source (LongTensor): index tensor ``(batch, len, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(batch, len, embedding_size)``
        """
        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, step=step)  # 最后一个模块是位置嵌入模块
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)

        if self.dropout_p > 0:
            return self.dropout(source)
        else:
            return source

    def update_dropout(self, dropout):
        self.dropout.p = dropout


# Some utilitary functions for pretrained embeddings


def read_embeddings(path, skip_lines=0, filter_set=None):
    """
    Read an embeddings file in the glove format.
    """
    embs = dict()
    total_vectors_in_file = 0
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode("utf8").strip().split(" ")
            if len(l_split) == 2:
                continue
            total_vectors_in_file += 1
            if filter_set is not None and l_split[0] not in filter_set:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs, total_vectors_in_file


def calc_vocab_load_stats(vocab, loaded_embed_dict):
    matching_count = len(set(vocab.ids_to_tokens) & set(loaded_embed_dict.keys()))
    missing_count = len(vocab) - matching_count
    percent_matching = matching_count / len(vocab) * 100
    return matching_count, missing_count, percent_matching


def convert_to_torch_tensor(word_to_float_list_dict, vocab):
    dim = len(next(iter(word_to_float_list_dict.values())))
    tensor = torch.zeros((len(vocab), dim))
    for word, values in word_to_float_list_dict.items():
        tensor[vocab.tokens_to_ids[word]] = torch.Tensor(values)
    return tensor


def prepare_pretrained_embeddings(opt, vocabs):
    if all([opt.both_embeddings is None, opt.src_embeddings is None, opt.tgt_embeddings is None,]):
        return

    assert (opt.save_data), "-save_data is required when using pretrained embeddings."

    vocs = []
    for side in ["src", "tgt"]:
        vocab = vocabs[side]
        vocs.append(vocab)
    enc_vocab, dec_vocab = vocs

    skip_lines = 1 if opt.embeddings_type == "word2vec" else 0
    if opt.both_embeddings is not None:
        set_of_src_and_tgt_vocab = set(enc_vocab.ids_to_tokens) | set(dec_vocab.ids_to_tokens)
        logger.info("Reading encoder and decoder embeddings from {}".format(opt.both_embeddings))
        src_vectors, total_vec_count = read_embeddings(opt.both_embeddings, skip_lines, set_of_src_and_tgt_vocab)
        tgt_vectors = src_vectors
        logger.info("\tFound {} total vectors in file".format(total_vec_count))
    else:
        if opt.src_embeddings is not None:
            logger.info("Reading encoder embeddings from {}".format(opt.src_embeddings))
            src_vectors, total_vec_count = read_embeddings(
                opt.src_embeddings, skip_lines, filter_set=set(enc_vocab.ids_to_tokens)
            )
            logger.info("\tFound {} total vectors in file.".format(total_vec_count))
        else:
            src_vectors = None

        if opt.tgt_embeddings is not None:
            logger.info("Reading decoder embeddings from {}".format(opt.tgt_embeddings))
            tgt_vectors, total_vec_count = read_embeddings(
                opt.tgt_embeddings, skip_lines, filter_set=set(dec_vocab.ids_to_tokens)
            )
            logger.info("\tFound {} total vectors in file".format(total_vec_count))
        else:
            tgt_vectors = None

    logger.info("After filtering to vectors in vocab:")
    if opt.src_embeddings is not None or opt.both_embeddings is not None:
        logger.info(
            "\t* enc: %d match, %d missing, (%.2f%%)"
            % calc_vocab_load_stats(enc_vocab, src_vectors)
        )
    if opt.tgt_embeddings is not None or opt.both_embeddings is not None:
        logger.info(
            "\t* dec: %d match, %d missing, (%.2f%%)"
            % calc_vocab_load_stats(dec_vocab, tgt_vectors)
        )

    # Write to file
    enc_output_file = opt.save_data + ".enc_embeddings.pt"
    dec_output_file = opt.save_data + ".dec_embeddings.pt"
    if opt.src_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\nSaving encoder embeddings as:\n\t* enc: %s" % enc_output_file)
        torch.save(convert_to_torch_tensor(src_vectors, enc_vocab), enc_output_file)
        # set the opt in place
        opt.pre_word_vecs_enc = enc_output_file
    if opt.tgt_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\nSaving decoder embeddings as:\n\t* dec: %s" % dec_output_file)
        torch.save(convert_to_torch_tensor(tgt_vectors, dec_vocab), dec_output_file)
        # set the opt in place
        opt.pre_word_vecs_dec = dec_output_file
