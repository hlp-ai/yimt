import configargparse as cfargparse
import os
import torch

import onmt.opts as opts
from onmt.utils.logging import logger
from onmt.constants import CorpusName, ModelTask
from onmt.transforms import AVAILABLE_TRANSFORMS


class DataOptsCheckerMixin(object):
    """Checker with methods for validate data related options."""

    @staticmethod
    def _validate_file(file_path, info):
        """Check `file_path` is valid or raise `IOError`."""
        if not os.path.isfile(file_path):
            raise IOError(f"Please check path of your {info} file!")

    @classmethod
    def _validate_data(cls, opt):
        """Parse corpora specified in data field of YAML file."""
        import yaml

        default_transforms = opt.transforms
        if len(default_transforms) != 0:
            logger.info(f"Default transforms: {default_transforms}.")
        corpora = yaml.safe_load(opt.data)

        for cname, corpus in corpora.items():
            # Check Transforms
            _transforms = corpus.get("transforms", None)
            if _transforms is None:
                logger.info(
                    f"Missing transforms field for {cname} data, "
                    f"set to default: {default_transforms}."
                )
                corpus["transforms"] = default_transforms
            # Check path
            path_src = corpus.get("path_src", None)
            path_tgt = corpus.get("path_tgt", None)
            path_txt = corpus.get("path_txt", None)
            if path_src is None and path_txt is None:
                raise ValueError(
                    f"Corpus {cname} src/txt path is required."
                    "tgt path is also required for non language"
                    " modeling tasks."
                )
            else:
                opt.data_task = ModelTask.SEQ2SEQ
                if path_tgt is None:
                    raise ValueError("path_tgt is None, it should be set")

                if path_src is not None:
                    cls._validate_file(path_src, info=f"{cname}/path_src")
                if path_txt is not None:
                    cls._validate_file(path_txt, info=f"{cname}/path_txt")
                if path_tgt is not None:
                    cls._validate_file(path_tgt, info=f"{cname}/path_tgt")

            # Check weight
            weight = corpus.get("weight", None)
            if weight is None:
                if cname != CorpusName.VALID:
                    logger.warning(
                        f"Corpus {cname}'s weight should be given."
                        " We default it to 1 for you."
                    )
                corpus["weight"] = 1

        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opt.data = corpora

    @classmethod
    def _validate_transforms_opts(cls, opt):
        """Check options used by transforms."""
        for name, transform_cls in AVAILABLE_TRANSFORMS.items():
            if name in opt._all_transform:
                transform_cls._validate_options(opt)

    @classmethod
    def _get_all_transform(cls, opt):
        """Should only called after `_validate_data`."""
        all_transforms = set(opt.transforms)
        for cname, corpus in opt.data.items():
            _transforms = set(corpus["transforms"])
            if len(_transforms) != 0:
                all_transforms.update(_transforms)

        opt._all_transform = all_transforms

    @classmethod
    def _get_all_transform_translate(cls, opt):
        opt._all_transform = opt.transforms

    @classmethod
    def _validate_vocab_opts(cls, opt):
        """Check options relate to vocab."""

        # validation when train:
        cls._validate_file(opt.src_vocab, info="src vocab")
        if not opt.share_vocab:
            cls._validate_file(opt.tgt_vocab, info="tgt vocab")

        # Check embeddings stuff
        if opt.both_embeddings is not None:
            assert (
                opt.src_embeddings is None and opt.tgt_embeddings is None
            ), "You don't need -src_embeddings or -tgt_embeddings \
                if -both_embeddings is set."

        if any(
            [
                opt.both_embeddings is not None,
                opt.src_embeddings is not None,
                opt.tgt_embeddings is not None,
            ]
        ):
            assert (
                opt.embeddings_type is not None
            ), "You need to specify an -embedding_type!"
            assert (
                opt.save_data
            ), "-save_data should be set if use \
                pretrained embeddings."

    @classmethod
    def validate_prepare_opts(cls, opt):
        """Validate all options relate to prepare (data/transform/vocab)."""
        cls._validate_data(opt)
        cls._get_all_transform(opt)
        cls._validate_transforms_opts(opt)
        cls._validate_vocab_opts(opt)


class ArgumentParser(cfargparse.ArgumentParser, DataOptsCheckerMixin):
    """OpenNMT option parser powered with option check methods."""

    def __init__(
        self,
        config_file_parser_class=cfargparse.YAMLConfigFileParser,
        formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
        **kwargs,
    ):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs,
        )

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def update_model_opts(cls, model_opt):
        if model_opt.word_vec_size > 0:  # 同时设置src和tgt
            model_opt.src_word_vec_size = model_opt.word_vec_size
            model_opt.tgt_word_vec_size = model_opt.word_vec_size

        if model_opt.layers > 0:  # 同时设置enc和dec层数
            model_opt.enc_layers = model_opt.layers
            model_opt.dec_layers = model_opt.layers

        if model_opt.hidden_size > 0:  # 同时设置enc和dec大小
            model_opt.enc_hid_size = model_opt.hidden_size
            model_opt.dec_hid_size = model_opt.hidden_size

    @classmethod
    def validate_model_opts(cls, model_opt):
        # encoder and decoder should be same sizes
        same_size = model_opt.enc_hid_size == model_opt.dec_hid_size
        assert same_size, "The encoder and decoder must be the same size for now"  # TODO: transformer也要这样吗

    @classmethod
    def ckpt_model_opts(cls, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = cls.defaults(opts.model_opts)
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

    @classmethod
    def validate_train_opts(cls, opt):
        if torch.cuda.is_available() and not opt.gpu_ranks:
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        if opt.world_size < len(opt.gpu_ranks):
            raise AssertionError(
                "parameter counts of -gpu_ranks must be less or equal than -world_size."
            )
        if opt.world_size == len(opt.gpu_ranks) and min(opt.gpu_ranks) > 0:
            raise AssertionError(
                "-gpu_ranks should have master(=0) rank unless -world_size is greater than len(gpu_ranks)."
            )

        assert len(opt.dropout) == len(
            opt.dropout_steps
        ), "Number of dropout values must match accum_steps values"

        assert len(opt.attention_dropout) == len(
            opt.dropout_steps
        ), "Number of attention_dropout values must match accum_steps values"

        assert len(opt.accum_count) == len(
            opt.accum_steps
        ), "Number of accum_count values must match number of accum_steps"

        if opt.update_vocab:
            assert opt.train_from, "-update_vocab needs -train_from option"
            assert opt.reset_optim in [
                "states",
                "all",
            ], '-update_vocab needs -reset_optim "states" or "all"'

    @classmethod
    def validate_translate_opts_dynamic(cls, opt):
        # It comes from training
        # TODO: needs to be added as inference opt
        opt.share_vocab = False
