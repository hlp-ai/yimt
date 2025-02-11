"""Module defining encoders."""
from onmt.encoders.transformer import TransformerEncoder


str2enc = {
    "transformer": TransformerEncoder,
}

