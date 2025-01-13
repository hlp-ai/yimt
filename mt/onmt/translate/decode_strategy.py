import torch
from copy import deepcopy

from onmt.utils.misc import tile


class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
      pad (int): Magic integer in output vocab.
      bos (int): Magic integer in output vocab.
      eos (int): Magic integer in output vocab.
      unk (int): Magic integer in output vocab.
      start (int): Magic integer in output vocab.
      batch_size (int): Current batch size.
      parallel_paths (int): Decoding strategies like beam search
        use parallel paths. Each batch is repeated ``parallel_paths``
        times in relevant state tensors.
      min_length (int): Shortest acceptable generation, not counting
        begin-of-sentence or end-of-sentence.
      max_length (int): Longest acceptable sequence, not counting
        begin-of-sentence (presumably there has been no EOS
        yet if max_length is used as a cutoff).
      ban_unk_token (Boolean): Whether unk token is forbidden
      return_attention (bool): Whether to work with attention too. If this
        is true, it is assumed that the decoder is attentional.

    Attributes:
      pad (int): See above.
      bos (int): See above.
      eos (int): See above.
      unk (int): See above.
      start (int): See above.
      predictions (list[list[LongTensor]]): For each batch, holds a
        list of beam prediction sequences.
      scores (list[list[FloatTensor]]): For each batch, holds a
        list of scores.
      attention (list[list[FloatTensor or list[]]]): For each
        batch, holds a list of attention sequence tensors
        (or empty lists) having shape ``(step, inp_seq_len)`` where
        ``inp_seq_len`` is the length of the sample (not the max
        length of all inp seqs).
      alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
        This sequence grows in the ``step`` axis on each call to
        :func:``advance()``.
      is_finished (ByteTensor or NoneType): Shape ``(B, parallel_paths)``.
        Initialized to ``None``.
      alive_attn (FloatTensor or NoneType): If tensor, shape is
        ``(B x parallel_paths, step, inp_seq_len)``, where ``inp_seq_len``
        is the (max) length of the input sequence.
      target_prefix (LongTensor or NoneType): If tensor, shape is
        ``(B x parallel_paths, prefix_seq_len)``, where ``prefix_seq_len``
        is the (max) length of the pre-fixed prediction.
      min_length (int): See above.
      max_length (int): See above.
      ban_unk_token (Boolean): See above.
      return_attention (bool): See above.
      done (bool): See above."""

    def __init__(
        self,
        pad,
        bos,
        eos,
        unk,
        start,
        batch_size,
        parallel_paths,
        global_scorer,
        min_length,
        return_attention,
        max_length,
        ban_unk_token,
    ):
        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.start = start

        self.batch_size = batch_size
        self.parallel_paths = parallel_paths  # beam_size
        self.global_scorer = global_scorer

        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]
        self.hypotheses = [[] for _ in range(batch_size)]

        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.ban_unk_token = ban_unk_token

        self.return_attention = return_attention

        self.done = False

    def get_device_from_enc_out(self, enc_out):
        if isinstance(enc_out, tuple):
            mb_device = enc_out[0].device
        else:
            mb_device = enc_out.device
        return mb_device

    def initialize_tile(self, enc_out, src_len, target_prefix=None):
        def fn_map_state(state, dim=0):
            return tile(state, self.beam_size, dim=dim)

        if isinstance(enc_out, tuple):
            enc_out = tuple(tile(x, self.beam_size, dim=0) for x in enc_out)
        elif enc_out is not None:
            enc_out = tile(enc_out, self.beam_size, dim=0)

        self.src_len = tile(src_len, self.beam_size)

        if target_prefix is not None:
            target_prefix = tile(target_prefix, self.beam_size, dim=0)

        return fn_map_state, enc_out, target_prefix

    def initialize(self, device=None, target_prefix=None):
        """DecodeStrategy subclasses should override :func:`initialize()`.

        `initialize` should be called before all actions.
        used to prepare necessary ingredients for decode."""
        if device is None:
            device = torch.device("cpu")

        # Here we set the decoder to start with self.start (BOS or EOS)
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1],
            self.start,
            dtype=torch.long,
            device=device,
        )

        self.is_finished_list = [[False for _ in range(self.parallel_paths)] for _ in range(self.batch_size)]

        if target_prefix is not None:
            batch_size, seq_len, n_feats = target_prefix.size()
            assert (
                batch_size == self.batch_size * self.parallel_paths
            ), "forced target_prefix should've extend to same number of path!"

            target_prefix_words = target_prefix[:, :, 0]  # no features
            target_prefix = target_prefix_words[:, 1:]  # remove bos

            # fix length constraint and remove eos from count
            prefix_non_pad = target_prefix.ne(self.pad).sum(dim=-1).tolist()
            self.max_length += max(prefix_non_pad) - 1
            self.min_length += min(prefix_non_pad) - 1

        self.target_prefix = target_prefix  # NOTE: forced prefix words

        return None

    def __len__(self):
        """生成序列长"""
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -65504  # -1e20，将结束符置为几乎不可能

    def ensure_unk_removed(self, log_probs):
        if self.ban_unk_token:
            log_probs[:, self.unk] = -65504  # -1e20，将UNK置为几乎不可能

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished_list = [
                [True for _ in range(self.parallel_paths)]
                for _ in range(len(self.is_finished_list))
            ]  # 设置为结束

    def target_prefixing(self, log_probs):
        """Fix the first part of predictions with `self.target_prefix`.

        Args:
        log_probs (FloatTensor): logits of size ``(B, vocab_size)``.

        Returns:
        log_probs (FloatTensor): modified logits in ``(B, vocab_size)``.
        """
        _B, vocab_size = log_probs.size()
        step = len(self)
        if self.target_prefix is not None and step <= self.target_prefix.size(1):
            pick_idx = self.target_prefix[:, step - 1].tolist()  # (B)
            pick_coo = [
                [path_i, pick]
                for path_i, pick in enumerate(pick_idx)
                if pick not in [self.eos, self.pad]
            ]
            mask_pathid = [
                path_i
                for path_i, pick in enumerate(pick_idx)
                if pick in [self.eos, self.pad]
            ]
            if len(pick_coo) > 0:
                pick_coo = torch.tensor(pick_coo).to(self.target_prefix)
                pick_fill_value = torch.ones([pick_coo.size(0)], dtype=log_probs.dtype)
                # pickups: Tensor where specified index were set to 1, others 0
                pickups = torch.sparse_coo_tensor(
                    pick_coo.t(),
                    pick_fill_value,
                    size=log_probs.size(),
                    device=log_probs.device,
                ).to_dense()
                # dropdowns: opposite of pickups, 1 for those shouldn't pick
                dropdowns = torch.ones_like(pickups) - pickups
                if len(mask_pathid) > 0:
                    path_mask = torch.zeros(_B).to(self.target_prefix)
                    path_mask[mask_pathid] = 1
                    path_mask = path_mask.unsqueeze(1).to(dtype=bool)
                    dropdowns = dropdowns.masked_fill(path_mask, 0)
                # Minus dropdowns to log_probs making probabilities of
                # unspecified index close to 0
                log_probs -= 10000 * dropdowns
        return log_probs

    def maybe_update_target_prefix(self, select_index):
        """We update / reorder `target_prefix` for alive path."""
        if self.target_prefix is None:
            return
        # prediction step have surpass length of given target_prefix,
        # no need to further change this attr
        if len(self) > self.target_prefix.size(1):
            return
        self.target_prefix = self.target_prefix[select_index]

    def advance(self, log_probs, attn):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """
        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """
        raise NotImplementedError()
