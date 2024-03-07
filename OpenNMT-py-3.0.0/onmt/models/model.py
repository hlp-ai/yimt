""" Onmt NMT Model base class definition """
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.
    """

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for input this will be a padded `LongTensor`
                of size ``(batch, len, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(batch, tgt_len, features)``.
            src_len(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(batch, tgt_len, hidden)``
            * dictionary attention dists of ``(batch, tgt_len, src_len)``
        """
        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError


class NMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):
        dec_in = tgt[:, :-1, :]  # exclude last target from input

        enc_out, enc_final_hs, src_len = self.encoder(src, src_len)

        if not bptt:
            # RNN uses enc_final_hs
            # CNN uses enc_out and enc_final_hs
            # transformer uses src
            self.decoder.init_state(src, enc_out, enc_final_hs)
        dec_out, attns = self.decoder(dec_in, enc_out,
                                      src_len=src_len,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout, attention_dropout):
        self.encoder.update_dropout(dropout, attention_dropout)
        self.decoder.update_dropout(dropout, attention_dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec
