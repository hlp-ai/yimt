""" Multi-Head Attention module """
import torch
import torch.nn as nn
from math import log, sqrt
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import skip_init
from torch.distributed import all_reduce
from importlib import import_module


# Help functions to split model dim per head


def shape(x: Tensor, dim_per_head: int) -> Tensor:
    """
    Projection.
    [batchsize x length x modeldim]
    -> [batchsize x heads x length x dimperhead]
    """
    x_0, x_1, _ = x.size()
    return x.view(x_0, x_1, -1, dim_per_head).transpose(1, 2)


def unshape(x: Tensor) -> Tensor:
    """
    Compute context.
    [batchsize x heads x length x dimperhead]
    -> [batchsize x length x modeldim]
    """
    x_0, x_1, _, x_3 = x.size()
    return x.transpose(1, 2).contiguous().view(x_0, -1, x_1 * x_3)


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
       attn_type: "self" or "context"
    """

    def __init__(
        self,
        head_count: int,
        model_dim: int,
        dropout: float = 0.1,
        is_decoder: bool = True,
        attn_type: str = None,
        self_attn_type: str = None,
        add_qkvbias=False,
        use_ckpting=[],
        parallel_gpu=1,
    ) -> None:
        assert (model_dim % head_count == 0), "Model dimension must be divisible by the number of heads"
        self.dim_per_head = model_dim // head_count
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.parallel_gpu = parallel_gpu

        assert (model_dim % parallel_gpu == 0), "Model dimension must be divisible by the number of partitions"
        self.linear_keys = skip_init(
            nn.Linear,
            in_features=model_dim,
            out_features=model_dim // parallel_gpu,
            bias=add_qkvbias,
        )
        self.linear_values = skip_init(
            nn.Linear,
            in_features=model_dim,
            out_features=model_dim // parallel_gpu,
            bias=add_qkvbias,
        )

        self.linear_query = skip_init(
            nn.Linear,
            in_features=model_dim,
            out_features=model_dim // parallel_gpu,
            bias=add_qkvbias,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.final_linear = skip_init(
            nn.Linear,
            in_features=model_dim // parallel_gpu,
            out_features=model_dim,
            bias=add_qkvbias,
        )
        self.is_decoder = is_decoder
        self.attn_type = attn_type
        self.self_attn_type = self_attn_type
        self.layer_cache = (
            False,
            {"keys": torch.tensor([]), "values": torch.tensor([])},
        )

        # self.cos = None
        # self.sin = None
        self.rotary_interleave = None

        self.maybe_ckpt = checkpoint if "mha" in use_ckpting else lambda f, x: f(x)

        try:
            flash_pack = import_module("flash_attn")
            if (
                hasattr(flash_pack, "flash_attn_func")
                and torch.cuda.get_device_capability()[0] >= 8
            ):
                self.flash_attn_func = getattr(flash_pack, "flash_attn_func")
                self.flash_attn_with_kvcache = getattr(flash_pack, "flash_attn_with_kvcache")
                self.flash2 = True
            else:
                self.flash2 = False
        except ImportError:
            self.flash2 = False

        print("flash-attn:", self.flash2)

    def update_dropout(self, dropout: float) -> None:
        self.dropout.p = dropout
        self.dropout_p = dropout

    def forward(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        mask: Optional[Tensor] = None,
        sliding_window: Optional[int] = 0,
        step: Optional[int] = 0,
        return_attn: Optional[bool] = False,
        self_attn_type: str = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the context vector and the attention vectors.

        Args:
           key (Tensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (Tensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (Tensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           step (int): decoding step (used for Rotary embedding)
        Returns:
           (Tensor, Tensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """
        # 1) Project key, value, and query.
        # as a reminder at training layer_cache[0] remains False
        key_pad_mask = self.layer_cache[1].get("key_pad_mask", None)
        if self.layer_cache[0]:  # 有层KV缓存，推理模式
            # Retrieve keys and values from the KV cache (decoding mode only).
            if self.attn_type == "self":  # 自注意力
                query, key, value = (self.linear_query(query), self.linear_keys(query), self.linear_values(query),)

                query = shape(query, self.dim_per_head)
                key = shape(key, self.dim_per_head)
                value = shape(value, self.dim_per_head)
                start_pos = step
                seqlen = query.size(2)

                if (
                    step == 0
                    or not self.flash2
                    or self.self_attn_type != "scaled-dot-flash"
                    # or self.max_relative_positions not in [0, -1]
                    or query.size(0) > 128
                    or query.dtype != torch.float16
                ):
                    if self.layer_cache[1]["keys"].numel() != 0:
                        key = torch.cat((self.layer_cache[1]["keys"], key), dim=2)
                        value = torch.cat((self.layer_cache[1]["values"], value), dim=2)
                        if sliding_window > 0 and key.size(2) > sliding_window:
                            key = key[:, :, 1:, :]
                            value = value[:, :, 1:, :]

                    self.layer_cache[1]["keys"] = key
                    self.layer_cache[1]["values"] = value

                else:
                    if start_pos >= self.layer_cache[1]["keys"].size(2):
                        self.layer_cache[1]["keys"] = torch.cat(
                            [
                                self.layer_cache[1]["keys"],
                                torch.zeros(
                                    self.layer_cache[1]["keys"].shape[:-2]
                                    + (32,)
                                    + self.layer_cache[1]["keys"].shape[-1:],
                                    device=query.device,
                                ).half(),
                            ],
                            dim=-2,
                        )
                        self.layer_cache[1]["values"] = torch.cat(
                            [
                                self.layer_cache[1]["values"],
                                torch.zeros(
                                    self.layer_cache[1]["values"].shape[:-2]
                                    + (32,)
                                    + self.layer_cache[1]["values"].shape[-1:],
                                    device=query.device,
                                ).half(),
                            ],
                            dim=-2,
                        )

                    if sliding_window > 0 and key.size(2) > sliding_window:
                        self.layer_cache[1]["keys"] = self.layer_cache[1]["keys"][:, :, 1:, :]
                        self.layer_cache[1]["values"] = self.layer_cache[1]["values"][:, :, 1:, :]
                    context = self.flash_attn_with_kvcache(
                        query.transpose(1, 2),
                        self.layer_cache[1]["keys"].transpose(1, 2),
                        self.layer_cache[1]["values"].transpose(1, 2),
                        key.transpose(1, 2),
                        value.transpose(1, 2),
                        rotary_cos=None,
                        rotary_sin=None,
                        cache_seqlens=step,
                        rotary_interleaved=self.rotary_interleave,
                    ).transpose(1, 2)
                    attn_output = self.final_linear(unshape(context))
                    if self.parallel_gpu > 1:
                        all_reduce(attn_output)
                    return attn_output, None

            elif self.attn_type == "context":  # 上下文注意力
                query = self.linear_query(query)
                query = shape(query, self.dim_per_head)
                if self.layer_cache[1]["keys"].numel() == 0:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key, self.dim_per_head)
                    value = shape(value, self.dim_per_head)
                else:
                    key, value = (self.layer_cache[1]["keys"], self.layer_cache[1]["values"],)
                self.layer_cache[1]["keys"] = key
                self.layer_cache[1]["values"] = value

            if key_pad_mask is not None:
                # Increase the cached key pad mask by concatenation.
                # For decoding only.
                if step > 0:
                    y = torch.zeros(
                        (key_pad_mask.size(0), key_pad_mask.size(1), 1),
                        dtype=torch.bool,
                        device=key_pad_mask.device,
                    )
                    self.layer_cache[1]["key_pad_mask"] = torch.cat((key_pad_mask, y), 2)
                    key_pad_mask = self.layer_cache[1]["key_pad_mask"]
        else:  # 无层KV缓存，训练模式
            # Retrieve keys and values from linear layers (training mode).
            key = self.maybe_ckpt(self.linear_keys, key)
            value = self.maybe_ckpt(self.linear_values, value)
            query = self.maybe_ckpt(self.linear_query, query)

            key = shape(key, self.dim_per_head)
            value = shape(value, self.dim_per_head)
            query = shape(query, self.dim_per_head)

        b, h, l, d = key.size()

        # 2) When standard pos. enc. or rotary, use flash attention

        # Ultimately flashv2 will be part of pytorch https://github.com/pytorch/pytorch/pull/105602
        # In the meantime: if vanilla tranformer or Rotary embeddings (not rel_pos, not alibi)
        # then use flash2 if seq len > 256 otherwise use xtransformer from pt2 uptream
        flash2 = (
            self.flash2
            and l > 256  # https://github.com/Dao-AILab/flash-attention/issues/591
        )

        if (
            # self.max_relative_positions in [-1, 0]
            not return_attn
            and query.device != torch.device("cpu")
            and self.self_attn_type == "scaled-dot-flash"
        ):
            # Apply flash2 attention.
            causal = self.is_decoder and self.attn_type == "self" and mask is not None
            if self.is_decoder and self.attn_type == "self" and flash2:  # 使用flash-attn
                if causal:
                    window_size = ((-1, -1) if sliding_window == 0 else (sliding_window, 0))
                else:
                    window_size = (-1, -1)
                attn_output = self.flash_attn_func(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    dropout_p=self.dropout_p,
                    causal=causal,
                    window_size=window_size,
                ).transpose(1, 2)
            else:
                # 使用nn.scaled-dot-flash/scaled_dot_product_attention
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
                    attn_output = scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        ~mask if mask is not None else None,
                        self.dropout_p,
                        is_causal=causal,
                    )
            attn = None

        else:  # 自定义scaled-dot attn
            query /= sqrt(self.dim_per_head)
            # batch x num_heads x query_len x key_len
            scores = torch.matmul(query, key.transpose(2, 3))

            scores = scores.float()
            if key_pad_mask is not None and mask is None:
                mask = key_pad_mask.unsqueeze(1)

            if mask is not None:
                # not 100% necessary but expand to nb of heads
                mask = mask.expand(-1, self.head_count // self.parallel_gpu, -1, -1)
                # now mask and scores have the same shape
                scores = scores.masked_fill(mask, -1e18)

            # 3) Apply attention dropout and compute context vectors.
            attn = self.softmax(scores).to(query.dtype)
            drop_attn = self.dropout(attn) if self.dropout_p > 0 else attn

            attn_output = torch.matmul(drop_attn, value)

        context = unshape(attn_output)
        if key_pad_mask is not None:
            if key_pad_mask.size(0) > 1 and context.size(1) > 1:
                x = key_pad_mask.squeeze(1).unsqueeze(2).expand(-1, -1, context.size(2))
                context = context.masked_fill(x, 0)

        if self.layer_cache[0]:
            attn_output = self.final_linear(context)
        else:
            attn_output = self.maybe_ckpt(self.final_linear, context)

        if self.parallel_gpu > 1:
            all_reduce(attn_output)

        return attn_output, attn
