"""Module defining decoders."""
from onmt.decoders.transformer import TransformerDecoder

str2dec = {
    "transformer": TransformerDecoder,
}
