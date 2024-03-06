"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase
from onmt.decoders.transformer import TransformerDecoder, TransformerLMDecoder


str2dec = {"transformer": TransformerDecoder,
           "transformer_lm": TransformerLMDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "str2dec", "TransformerLMDecoder"]
