"""Transforms relate to tokenization/subword."""
from onmt.transforms import register_transform
from .transform import Transform, ObservableStats
from onmt.constants import DefaultTokens


class TokenizerTransform(Transform):
    """Tokenizer transform abstract class."""

    def __init__(self, opts):
        """Initialize necessary options for Tokenizer."""
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to Subword."""
        # Sharing options among `TokenizerTransform`s, same name conflict in
        # this scope will be resolved by remove previous occurrence in parser
        group = parser.add_argument_group(
            "Transform/Subword/Common",
            conflict_handler="resolve",
            description=".. Attention:: Common options shared by all subword transforms. "  # noqa: E501
            "Including options for indicate subword model path, "
            "`Subword Regularization <https://arxiv.org/abs/1804.10959>`_"
            "/`BPE-Dropout <https://arxiv.org/abs/1910.13267>`_, "
            "and `Vocabulary Restriction <https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt>`__.",  # noqa: E501
        )  # noqa: E501
        group.add(
            "-src_subword_model",
            "--src_subword_model",
            help="Path of subword model for src (or shared).",
        )
        group.add(
            "-tgt_subword_model",
            "--tgt_subword_model",
            help="Path of subword model for tgt.",
        )

        # subword regularization(or BPE dropout) options:
        group.add(
            "-src_subword_nbest",
            "--src_subword_nbest",
            type=int,
            default=1,
            help="Number of candidates in subword regularization. "
            "Valid for unigram sampling, "
            "invalid for BPE-dropout. "
            "(source side)",
        )
        group.add(
            "-tgt_subword_nbest",
            "--tgt_subword_nbest",
            type=int,
            default=1,
            help="Number of candidates in subword regularization. "
            "Valid for unigram sampling, "
            "invalid for BPE-dropout. "
            "(target side)",
        )
        group.add(
            "-src_subword_alpha",
            "--src_subword_alpha",
            type=float,
            default=0,
            help="Smoothing parameter for sentencepiece unigram "
            "sampling, and dropout probability for BPE-dropout. "
            "(source side)",
        )
        group.add(
            "-tgt_subword_alpha",
            "--tgt_subword_alpha",
            type=float,
            default=0,
            help="Smoothing parameter for sentencepiece unigram "
            "sampling, and dropout probability for BPE-dropout. "
            "(target side)",
        )

        # subword vocabulary restriction options:
        group.add(
            "-src_subword_vocab",
            "--src_subword_vocab",
            type=str,
            default="",
            help="Path to the vocabulary file for src subword. "
            "Format: <word>\t<count> per line.",
        )
        group.add(
            "-tgt_subword_vocab",
            "--tgt_subword_vocab",
            type=str,
            default="",
            help="Path to the vocabulary file for tgt subword. "
            "Format: <word>\t<count> per line.",
        )
        group.add(
            "-src_vocab_threshold",
            "--src_vocab_threshold",
            type=int,
            default=0,
            help="Only produce src subword in src_subword_vocab with "
            " frequency >= src_vocab_threshold.",
        )
        group.add(
            "-tgt_vocab_threshold",
            "--tgt_vocab_threshold",
            type=int,
            default=0,
            help="Only produce tgt subword in tgt_subword_vocab with "
            " frequency >= tgt_vocab_threshold.",
        )

    @classmethod
    def _validate_options(cls, opts):
        """Extra checks for Subword options."""
        assert (
            0 <= opts.src_subword_alpha <= 1
        ), "src_subword_alpha should be in the range [0, 1]"
        assert (
            0 <= opts.tgt_subword_alpha <= 1
        ), "tgt_subword_alpha should be in the range [0, 1]"

    def _parse_opts(self):
        self.share_vocab = self.opts.share_vocab
        self.src_subword_model = self.opts.src_subword_model
        self.tgt_subword_model = self.opts.tgt_subword_model
        self.src_subword_nbest = self.opts.src_subword_nbest
        self.tgt_subword_nbest = self.opts.tgt_subword_nbest
        self.src_subword_alpha = self.opts.src_subword_alpha
        self.tgt_subword_alpha = self.opts.tgt_subword_alpha
        self.src_subword_vocab = self.opts.src_subword_vocab
        self.tgt_subword_vocab = self.opts.tgt_subword_vocab
        self.src_vocab_threshold = self.opts.src_vocab_threshold
        self.tgt_vocab_threshold = self.opts.tgt_vocab_threshold

    def _repr_args(self):
        """Return str represent key arguments for TokenizerTransform."""
        kwargs = {
            "share_vocab": self.share_vocab,
            "src_subword_model": self.src_subword_model,
            "tgt_subword_model": self.tgt_subword_model,
            "src_subword_alpha": self.src_subword_alpha,
            "tgt_subword_alpha": self.tgt_subword_alpha,
            "src_subword_vocab": self.src_subword_vocab,
            "tgt_subword_vocab": self.tgt_subword_vocab,
            "src_vocab_threshold": self.src_vocab_threshold,
            "tgt_vocab_threshold": self.tgt_vocab_threshold,
        }
        return ", ".join([f"{kw}={arg}" for kw, arg in kwargs.items()])

    def tokenize_string(self, string, side="src", is_train=False):
        raise NotImplementedError

    def _tokenize(self, tokens, side="src", is_train=False):
        """Tokenize a list of words."""
        # This method embeds a custom logic to correctly handle certain placeholders
        # in case the tokenizer doesn't preserve them.
        sentence = " ".join(tokens).replace(DefaultTokens.SEP, "\n")
        # Locate the end-of-sentence placeholders.
        sent_list = sentence.split(DefaultTokens.EOS)

        # Tokenize each sentence separately.
        segmented = []
        for _sentence in sent_list:
            # Locate the mask-before placeholders
            # (to zero-out the prompt loss during LM finetuning).
            _sentence_chunks = _sentence.split(DefaultTokens.MASK_BEFORE)

            # Tokenize each chunk separately and insert the padding token.
            # between each sequence of tokens.
            _sentence_tokens = []
            for _chunk in _sentence_chunks:
                _sentence_tokens += self.tokenize_string(_chunk, side, is_train) + [DefaultTokens.PAD]
            # Re-insert the eos token.
            segmented += _sentence_tokens[:-1] + [DefaultTokens.EOS]
        return segmented[:-1]

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply subword-based tokenenization to src & tgt."""
        src_out = self._tokenize(example["src"], "src", is_train)

        if example["tgt"] is not None:
            tgt_out = self._tokenize(example["tgt"], "tgt", is_train)
            if stats is not None:
                n_words = len(example["src"]) + len(example["tgt"])
                n_subwords = len(src_out) + len(tgt_out)
                stats.update(SubwordStats(n_subwords, n_words))
        else:
            tgt_out = None
            if stats is not None:
                n_words = len(example["src"])
                n_subwords = len(src_out)
                stats.update(SubwordStats(n_subwords, n_words))

        example["src"], example["tgt"] = src_out, tgt_out
        return example


class SubwordStats(ObservableStats):
    """Runing statistics for counting tokens before/after subword transform."""

    __slots__ = ["subwords", "words"]

    def __init__(self, subwords: int, words: int):
        self.subwords = subwords
        self.words = words

    def update(self, other: "SubwordStats"):
        self.subwords += other.subwords
        self.words += other.words

    def __str__(self) -> str:
        return "{}: {} -> {} tokens".format(self.name(), self.words, self.subwords)


@register_transform(name="sentencepiece")
class SentencePieceTransform(TokenizerTransform):
    """SentencePiece subword transform class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import sentencepiece as spm

        spm.set_random_generator_seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        import sentencepiece as spm

        load_src_model = spm.SentencePieceProcessor()
        load_src_model.Load(self.src_subword_model)

        _diff_vocab = (
            self.src_subword_vocab != self.tgt_subword_vocab
            or self.src_vocab_threshold != self.tgt_vocab_threshold
        )

        if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
            load_src_model.LoadVocabulary(self.src_subword_vocab, self.src_vocab_threshold)

        if self.share_vocab and not _diff_vocab:
            self.load_models = {"src": load_src_model, "tgt": load_src_model}
        else:
            load_tgt_model = spm.SentencePieceProcessor()
            load_tgt_model.Load(self.tgt_subword_model)
            if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
                load_tgt_model.LoadVocabulary(self.tgt_subword_vocab, self.tgt_vocab_threshold)
            self.load_models = {"src": load_src_model, "tgt": load_tgt_model}

    def tokenize_string(self, string, side="src", is_train=False):
        """Apply subword sampling or deterministic subwording"""
        sp_model = self.load_models[side]
        nbest_size = self.tgt_subword_nbest if side == "tgt" else self.src_subword_nbest
        if is_train is False or nbest_size in [0, 1]:
            # derterministic subwording
            tokens = sp_model.encode(string, out_type=str)
        else:
            # subword sampling when nbest_size > 1 or -1
            # alpha should be 0.0 < alpha < 1.0
            alpha = self.tgt_subword_alpha if side == "tgt" else self.src_subword_alpha
            tokens = sp_model.encode(
                string,
                out_type=str,
                enable_sampling=True,
                alpha=alpha,
                nbest_size=nbest_size,
            )
        return tokens

    def _detokenize(self, tokens, side="src"):
        """Apply SentencePiece Detokenizer"""
        sp_model = self.load_models[side]
        return sp_model.DecodePieces(tokens).replace("\n", DefaultTokens.SEP)

    def apply_reverse(self, translated):
        """Apply SentencePiece Detokenizer."""
        if isinstance(translated, list):
            return self._detokenize(translated, "tgt")
        else:
            return self._detokenize(translated.split(" "), "tgt")

    def _repr_args(self):
        """Return str represent key arguments for class."""
        kwargs_str = super()._repr_args()
        additional_str = "src_subword_nbest={}, tgt_subword_nbest={}".format(
            self.src_subword_nbest, self.tgt_subword_nbest
        )
        return kwargs_str + ", " + additional_str
