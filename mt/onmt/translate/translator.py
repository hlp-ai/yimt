#!/usr/bin/env python
""" Translator Class and builder """
import torch
from torch.nn.functional import log_softmax
import codecs
from time import time
from math import exp
from itertools import count, zip_longest
import onmt.model_builder
import onmt.decoders.ensemble
from onmt.constants import DefaultTokens
from onmt.translate.beam_search import BeamSearch
from onmt.translate.greedy_search import GreedySearch
from onmt.utils.misc import tile, set_random_seed, report_matrix
from onmt.transforms import TransformPipe


def build_translator(opt, device_id=0, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, "w+", "utf-8")

    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )

    vocabs, model, model_opt = load_test_model(opt, device_id)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
            model,
            vocabs,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger,
        )
    return translator


class Inference(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        vocabs (dict[str, Vocab]): A dict
            mapping each side's Vocab.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (float): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        replace_unk (bool): Replace unknown token.
        tgt_file_prefix (bool): Force the predictions begin with provided -tgt.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        vocabs,
        gpu=-1,
        n_best=1,
        min_length=0,
        max_length=100,
        max_length_ratio=1.5,
        ratio=0.0,
        beam_size=30,
        random_sampling_topk=0,
        random_sampling_topp=0.0,
        random_sampling_temp=1.0,
        replace_unk=False,
        ban_unk_token=False,
        tgt_file_prefix=False,
        phrase_table="",
        verbose=False,
        report_time=False,
        global_scorer=None,
        out_file=None,
        report_score=True,
        logger=None,
        seed=-1,
        with_score=False,
    ):
        self.model = model
        self.vocabs = vocabs
        self._tgt_vocab = vocabs["tgt"]
        self._tgt_eos_idx = vocabs["tgt"].lookup_token(DefaultTokens.EOS)
        self._tgt_pad_idx = vocabs["tgt"].lookup_token(DefaultTokens.PAD)
        self._tgt_bos_idx = vocabs["tgt"].lookup_token(DefaultTokens.BOS)
        self._tgt_unk_idx = vocabs["tgt"].lookup_token(DefaultTokens.UNK)
        self._tgt_start_with = vocabs["tgt"].lookup_token(vocabs["decoder_start_token"])
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = (torch.device("cuda", self._gpu) if self._use_cuda else torch.device("cpu"))

        self.n_best = n_best
        self.max_length = max_length
        self.max_length_ratio = max_length_ratio

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp

        self.min_length = min_length
        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.replace_unk = replace_unk

        self.tgt_file_prefix = tgt_file_prefix
        self.phrase_table = phrase_table
        self.verbose = verbose
        self.report_time = report_time

        self.global_scorer = global_scorer
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        set_random_seed(seed, self._use_cuda)
        self.with_score = with_score

    @classmethod
    def from_opt(
        cls,
        model,
        vocabs,
        opt,
        model_opt,
        global_scorer=None,
        out_file=None,
        report_score=True,
        logger=None,
    ):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            vocabs (dict[str, Vocab]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        return cls(
            model,
            vocabs,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            max_length_ratio=opt.max_length_ratio,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_topp=opt.random_sampling_topp,
            random_sampling_temp=opt.random_sampling_temp,
            replace_unk=opt.replace_unk,
            ban_unk_token=opt.ban_unk_token,
            tgt_file_prefix=opt.tgt_file_prefix,
            phrase_table=opt.phrase_table,
            verbose=opt.verbose,
            report_time=opt.report_time,
            global_scorer=global_scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger,
            seed=opt.seed,
            with_score=opt.with_score,
        )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, enc_out, src_len, batch_size, src):
        gs = [0] * batch_size
        glp = None
        return gs, glp

    def _translate(
        self,
        infer_iter,
        transform=None,
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            infer_iter: tensored batch iterator from DynamicDatasetIter
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        transform_pipe = (
            TransformPipe.build_from([transform[name] for name in transform])
            if transform
            else None
        )
        xlation_builder = onmt.translate.TranslationBuilder(
            self.vocabs,
            self.n_best,
            self.replace_unk,
            self.phrase_table,
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time()

        def _process_bucket(bucket_translations):
            bucket_scores = []
            bucket_predictions = []
            bucket_score = 0
            bucket_words = 0
            bucket_gold_score = 0
            bucket_gold_words = 0
            voc_src = self.vocabs["src"].ids_to_tokens

            # 恢复样本位置
            bucket_translations = sorted(bucket_translations, key=lambda x: x.ind_in_bucket)

            for trans in bucket_translations:
                bucket_scores += [trans.pred_scores[: self.n_best]]
                bucket_score += trans.pred_scores[0]
                bucket_words += len(trans.pred_sents[0])
                if "tgt" in batch.keys():
                    bucket_gold_score += trans.gold_score
                    bucket_gold_words += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred) for pred in trans.pred_sents[: self.n_best]]

                if transform_pipe is not None:
                    n_best_preds = transform_pipe.batch_apply_reverse(n_best_preds)

                bucket_predictions += [n_best_preds]

                if self.with_score:
                    n_best_scores = [score.item() for score in trans.pred_scores[: self.n_best]]
                    out_all = [
                        pred + "\t" + str(score)
                        for (pred, score) in zip(n_best_preds, n_best_scores)
                    ]
                    self.out_file.write("\n".join(out_all) + "\n")
                else:
                    self.out_file.write("\n".join(n_best_preds) + "\n")
                self.out_file.flush()

                if self.verbose:
                    srcs = [voc_src[tok] for tok in trans.src[: trans.srclen]]
                    sent_number = next(counter)
                    output = trans.log(sent_number, src_raw=srcs)
                    self._log(output)

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append(DefaultTokens.EOS)
                    attns = trans.attns[0].tolist()
                    srcs = [voc_src[tok] for tok in trans.src[: trans.srclen].tolist()]

                    output = report_matrix(srcs, preds, attns)
                    self._log(output)

                if align_debug:
                    tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    srcs = [voc_src[tok] for tok in trans.src[: trans.srclen].tolist()]

                    output = report_matrix(srcs, tgts, align)
                    self._log(output)

            return (
                bucket_scores,
                bucket_predictions,
                bucket_score,
                bucket_words,
                bucket_gold_score,
                bucket_gold_words,
            )

        bucket_translations = []
        prev_idx = 0

        for batch, bucket_idx in infer_iter:
            batch_data = self.translate_batch(batch, attn_debug)

            translations = xlation_builder.from_batch(batch_data)

            bucket_translations += translations

            if (
                not isinstance(infer_iter, list)  # 不是list，为什么判断？
                and len(bucket_translations) >= infer_iter.bucket_size  # 完成一个桶中所有batch
            ):
                bucket_idx += 1

            if bucket_idx != prev_idx:  # 下一个桶
                prev_idx = bucket_idx
                (
                    bucket_scores,
                    bucket_predictions,
                    bucket_score,
                    bucket_words,
                    bucket_gold_score,
                    bucket_gold_words,
                ) = _process_bucket(bucket_translations)
                all_scores += bucket_scores
                all_predictions += bucket_predictions
                pred_score_total += bucket_score
                pred_words_total += bucket_words
                gold_score_total += bucket_gold_score
                gold_words_total += bucket_gold_words
                bucket_translations = []

        if len(bucket_translations) > 0:
            (
                bucket_scores,
                bucket_predictions,
                bucket_score,
                bucket_words,
                bucket_gold_score,
                bucket_gold_words,
            ) = _process_bucket(bucket_translations)
            all_scores += bucket_scores
            all_predictions += bucket_predictions
            pred_score_total += bucket_score
            pred_words_total += bucket_words
            gold_score_total += bucket_gold_score
            gold_words_total += bucket_gold_words

        end_time = time()

        if self.report_score:
            msg = self._report_score("PRED", pred_score_total, len(all_scores))
            self._log(msg)
            if "tgt" in batch.keys() and not self.tgt_file_prefix:
                msg = self._report_score("GOLD", gold_score_total, len(all_scores))
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %.1f" % total_time)
            self._log(
                "Average translation time (ms): %.1f"
                % (total_time / len(all_predictions) * 1000)
            )
            self._log("Tokens per second: %.1f" % (pred_words_total / total_time))

        return all_scores, all_predictions

    def _report_score(self, name, score_total, nb_sentences):
        # In the case of length_penalty = none we report the total logprobs
        # divided by the number of sentence to get an approximation of the
        # per sentence logprob. We also return the corresponding ppl
        # When a length_penalty is used eg: "avg" or "wu" since logprobs
        # are normalized per token we report the per line per token logprob
        # and the corresponding "per word perplexity"
        if nb_sentences == 0:
            msg = "%s No translations" % (name,)
        else:
            score = score_total / nb_sentences
            try:
                ppl = exp(-score_total / nb_sentences)
            except OverflowError:
                ppl = float("inf")
            msg = "%s SCORE: %.4f, %s PPL: %.2f NB SENTENCES: %d" % (
                name,
                score,
                name,
                ppl,
                nb_sentences,
            )
        return msg

    def _decode_and_generate(
        self,
        decoder_in,
        enc_out,
        # batch,
        src_len,
        step=None,
        # batch_offset=None,
        return_attn=False,
    ):
        # Decoder forward, takes [batch, tgt_len, nfeats] as input
        # and [batch, src_len, hidden] as enc_out
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in,
            enc_out,
            src_len=src_len,
            step=step,
            return_attn=return_attn,
        )

        # Generator forward.
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None

        scores = self.model.generator(dec_out.squeeze(1))
        log_probs = log_softmax(scores, dim=-1)  # we keep float16 if FP16
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [batch_size, tgt_len, vocab ] when full sentence

        return log_probs, attn

    def translate_batch(self, batch, attn_debug):
        """Translate a batch of sentences."""
        raise NotImplementedError

    def report_results(
        self,
        gold_score,
        gold_log_probs,
        batch,
        batch_size,
        decode_strategy,
    ):
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": gold_score,
            "gold_log_probs": gold_log_probs,
        }

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["alignment"] = [[] for _ in range(batch_size)]
        return results


class Translator(Inference):

    def translate_batch(self, batch, attn_debug):
        """Translate a batch of sentences."""
        if self.max_length_ratio > 0:
            max_length = int(min(self.max_length, batch["src"].size(1) * self.max_length_ratio + 5))
        else:
            max_length = self.max_length

        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                # self._log("Decoding using GreedySearch")
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    batch_size=len(batch["srclen"]),
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # self._log("Decoding using BeamSearch")
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=len(batch["srclen"]),
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    return_attention=attn_debug or self.replace_unk,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )

            return self._translate_batch_with_strategy(batch, decode_strategy)

    def _run_encoder(self, batch):
        src = batch["src"]
        src_len = batch["srclen"]
        batch_size = len(batch["srclen"])

        enc_out, src_len = self.model.encoder(src, src_len)

        if src_len is None:
            assert not isinstance(enc_out, tuple), "Ensemble decoding only supported for text data"
            src_len = (torch.Tensor(batch_size).type_as(enc_out).long().fill_(enc_out.size(1)))
        return src, enc_out, src_len

    def _translate_batch_with_strategy(self, batch, decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        batch_size = len(batch["srclen"])

        # (1) 输入编码.
        src, enc_out, src_len = self._run_encoder(batch)

        # 解码器保留src作为状态
        self.model.decoder.init_state(src, enc_out)  # TODO: 有必要吗？

        # TODO: 黄金分数没有必要，只有占位代码，未来删除
        gold_score, gold_log_probs = self._gold_score(
            batch,
            enc_out,
            src_len,
            batch_size,
            src,
        )

        # (2) 准备decode_strategy. Possibly repeat src objects.
        target_prefix = batch["tgt"] if self.tgt_file_prefix else None
        (fn_map_state, enc_out) = decode_strategy.initialize(enc_out, src_len, target_prefix=target_prefix)

        # 初始化解码器状态
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        # (3) 逐步解码:
        for step in range(decode_strategy.max_length):
            # TODO: 取一步或长度为1序列作为输入，是否有问题？取前面所有预测序列作为输入？
            decoder_input = decode_strategy.current_predictions.view(-1, 1, 1)  # 前一步的预测，序列长度为1

            # 单步预测
            log_probs, attn = self._decode_and_generate(
                decoder_input,
                enc_out,
                #batch,
                src_len=decode_strategy.src_len,
                step=step,
                #batch_offset=decode_strategy.batch_offset,
                return_attn=decode_strategy.return_attention,
            )

            # 解码
            decode_strategy.advance(log_probs, attn)

            any_finished = any([any(sublist) for sublist in decode_strategy.is_finished_list])
            if any_finished:  # 有完成的
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(enc_out, tuple):
                    enc_out = tuple(x[select_indices] for x in enc_out)
                else:
                    enc_out = enc_out[select_indices]

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(lambda state, dim: state[select_indices])

        return self.report_results(
            gold_score,
            gold_log_probs,
            batch,
            batch_size,
            decode_strategy,
        )
