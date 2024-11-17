import torch.nn as nn


class DecoderBase(nn.Module):
    """Abstract class for decoders.
    """

    def __init__(self):
        super(DecoderBase, self).__init__()

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.

        Subclasses should override this method.
        """
        raise NotImplementedError
