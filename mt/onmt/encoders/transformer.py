"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
        add_ffnbias (bool): whether to add bias to the FF nn.Linear
        norm_eps (float): layer norm epsilon
        use_ckpting (List): layers for which we checkpoint for backward
        parallel_gpu (int): Number of gpu for tensor parallelism
    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        add_ffnbias=True,
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            is_decoder=False,
            attn_type="self",
            add_qkvbias=add_qkvbias,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            add_ffnbias,
            norm_eps,
            use_ckpting=use_ckpting,
            parallel_gpu=parallel_gpu,
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_in, mask):
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):
            * layer_out ``(batch_size, src_len, model_dim)``
        """
        norm_layer_in = self.layer_norm(layer_in)  # prenorm
        context, _ = self.self_attn(norm_layer_in, norm_layer_in, norm_layer_in, mask=mask)
        if self.dropout_p > 0:
            context = self.dropout(context)
        layer_out = context + layer_in
        layer_out = self.feed_forward(layer_out)

        return layer_out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * enc_out ``(batch_size, src_len, model_dim)``
        * src_len ``(batch_size)``
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        embeddings,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        add_ffnbias=True,
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
    ):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias,
                    add_ffnbias=add_ffnbias,
                    norm_eps=norm_eps,
                    use_ckpting=use_ckpting,
                    parallel_gpu=parallel_gpu,
                )
                for i in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0]
            if type(opt.attention_dropout) is list
            else opt.attention_dropout,
            embeddings,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            add_qkvbias=opt.add_qkvbias,
            add_ffnbias=opt.add_ffnbias,
            norm_eps=opt.norm_eps,
            use_ckpting=opt.use_ckpting,
            parallel_gpu=opt.world_size
            if opt.parallel_mode == "tensor_parallel"
            else 1,
        )

    def forward(self, src, src_len=None):
        """See :func:`EncoderBase.forward()`"""
        enc_out = self.embeddings(src)

        # 输入长度掩码
        mask = sequence_mask(src_len).unsqueeze(1).unsqueeze(1)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        # Padding mask is now (batch x 1 x slen x slen)
        # 1 to be expanded to number of heads in MHA
        # Run the forward pass of every layer of the tranformer.

        for layer in self.transformer:
            enc_out = layer(enc_out, mask)
        enc_out = self.layer_norm(enc_out)  # postnorm，和层的prenorm是否重叠?
        return enc_out, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
