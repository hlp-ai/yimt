"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
import torch
import torch.nn as nn
import onmt
from onmt.constants import  DefaultTokens

try:
    import ctranslate2
except ImportError:
    pass  # this is tested when importing for loading a LM


class LossCompute(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    accumulating multiple loss computations.

    Args:
        criterion (:obj:`nn. loss function`) : NLLoss or customed loss
        generator (:obj:`nn.Module`) :
        tgt_shift_index (int): 1 for NMT, 0 for LM
        vocab: target vocab (for copy attention score calculation)
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
    """

    def __init__(
        self,
        criterion,
        generator,
        tgt_shift_index=1,
        vocab=None,
    ):
        super(LossCompute, self).__init__()
        self.criterion = criterion
        self.generator = generator
        self.tgt_shift_index = tgt_shift_index
        self.vocab = vocab  # target vocab for copy_attn need

    @classmethod
    def from_opts(cls, opt, model, vocab, train=True):
        """
        Returns a subclass which wraps around an nn.Module subclass
        (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
        object passes relevant data to a Statistics object which handles
        training/validation logging.
        The Criterion and LossCompute options are triggered by opt settings.
        """
        device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

        padding_idx = vocab[DefaultTokens.PAD]
        unk_idx = vocab[DefaultTokens.UNK]

        tgt_shift_idx = 1 # if opt.model_task == ModelTask.SEQ2SEQ else 0

        criterion = nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            reduction="sum",
            label_smoothing=opt.label_smoothing,
        )

        compute = cls(
            criterion,
            model.generator,
            tgt_shift_index=tgt_shift_idx,
            vocab=vocab,
        )
        compute.to(device)

        return compute

    @property
    def padding_idx(self):
        return self.criterion.ignore_index


    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, batch, output, attns, trunc_start=0, trunc_size=None):
        """Compute the forward loss, supports truncated BPTT for long
        sequences by taking a range in the decoder output sequence to
        back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.
        Truncation is an approximate efficiency trick to relieve the
        memory required in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model ``(batch, tgt_len, hidden)``
          attns (dict) : dictionary of attention weights
              ``(batch, tgt_len, src_len)``
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """

        if trunc_size is None:
            trunc_size = batch["tgt"].size(1) - trunc_start
        # take into account here the tgt_shift_index (0 / 1 = LM/NMT)
        trunc_range = (trunc_start + self.tgt_shift_index, trunc_start + trunc_size)

        target = batch["tgt"][:, trunc_range[0] : trunc_range[1], :]
        output = output[:, trunc_start : trunc_range[1], :].contiguous()

        flat_tgt = target[:, :, 0].contiguous().view(-1)

        scores = self.generator(self._bottle(output))
        loss = self.criterion(scores.to(torch.float32), flat_tgt)

        n_sents = len(batch["srclen"]) if trunc_start == 0 else 0
        stats = self._stats(n_sents, loss.sum().item(), scores, flat_tgt)

        return loss, stats

    def _stats(self, bsz, loss, scores, target):
        """
        Args:
            loss (int): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        n_batchs = 1 if bsz else 0
        # in the case criterion reduction is None then we need
        # to sum the loss of each sentence in the batch
        return onmt.utils.Statistics(
            loss=loss,
            n_batchs=n_batchs,
            n_sents=bsz,
            n_words=num_non_padding,
            n_correct=num_correct,
        )
