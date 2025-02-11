"""Module defining encoders."""
import os
import importlib
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder


str2enc = {
    "transformer": TransformerEncoder,
}

__all__ = [
    "EncoderBase",
    "TransformerEncoder",
    "str2enc",
]


# Auto import python files in this directory
encoder_dir = os.path.dirname(__file__)
for file in os.listdir(encoder_dir):
    path = os.path.join(encoder_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        file_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("onmt.encoders." + file_name)
