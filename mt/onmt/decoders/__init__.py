"""Module defining decoders."""
import os
import importlib

from onmt.decoders.decoder import DecoderBase
from onmt.decoders.transformer import TransformerDecoder

str2dec = {
    "transformer": TransformerDecoder,
}

__all__ = [
    "TransformerDecoder",
    "str2dec",
]


# Auto import python files in this directory
decoder_dir = os.path.dirname(__file__)
for file in os.listdir(decoder_dir):
    path = os.path.join(decoder_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        file_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("onmt.decoders." + file_name)
