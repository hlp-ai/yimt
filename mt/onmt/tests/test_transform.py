"""Here come the tests for implemented transform."""
import unittest

import copy
import yaml
import math
from argparse import Namespace
from onmt.transforms import (
    get_transforms_cls,
    get_specials,
    make_transforms,
    TransformPipe,
)


class TestTransform(unittest.TestCase):
    def test_transform_register(self):
        builtin_transform = [
            "filtertoolong",
            "prefix",
            "sentencepiece",
        ]
        get_transforms_cls(builtin_transform)

    def test_transform_specials(self):
        transforms_cls = get_transforms_cls(["prefix"])
        corpora = yaml.safe_load(
            """
            trainset:
                path_src: ../../data/src-train.txt
                path_tgt: ../../data/tgt-train.txt
                transforms: ["prefix"]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """
        )
        opt = Namespace(data=corpora)
        specials = get_specials(opt, transforms_cls)
        specials_expected = {"src": ["｟_pf_src｠"], "tgt": ["｟_pf_tgt｠"]}
        self.assertEqual(specials, specials_expected)

    def test_transform_pipe(self):
        # 1. Init first transform in the pipe
        prefix_cls = get_transforms_cls(["prefix"])["prefix"]
        corpora = yaml.safe_load(
            """
            trainset:
                path_src: ../../data/src-train.txt
                path_tgt: ../../data/tgt-train.txt
                transforms: [prefix, filtertoolong]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """
        )
        opt = Namespace(data=corpora, seed=-1)
        prefix_transform = prefix_cls(opt)
        prefix_transform.warm_up()
        # 2. Init second transform in the pipe
        filter_cls = get_transforms_cls(["filtertoolong"])["filtertoolong"]
        opt = Namespace(src_seq_length=4, tgt_seq_length=4)
        filter_transform = filter_cls(opt)
        # 3. Sequential combine them into a transform pipe
        transform_pipe = TransformPipe.build_from([prefix_transform, filter_transform])
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # 4. apply transform pipe for example
        ex_after = transform_pipe.apply(copy.deepcopy(ex), corpus_name="trainset")
        # 5. example after the pipe exceed the length limit, thus filtered
        self.assertIsNone(ex_after)
        # 6. Transform statistics registed (here for filtertoolong)
        self.assertTrue(len(transform_pipe.statistics.observables) > 0)
        msg = transform_pipe.statistics.report()
        self.assertIsNotNone(msg)
        # 7. after report, statistics become empty as a fresh start
        self.assertTrue(len(transform_pipe.statistics.observables) == 0)


class TestMiscTransform(unittest.TestCase):
    def test_prefix(self):
        prefix_cls = get_transforms_cls(["prefix"])["prefix"]
        corpora = yaml.safe_load(
            """
            trainset:
                path_src: ../../data/src-train.txt
                path_tgt: ../../data/tgt-train.txt
                transforms: [prefix]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """
        )
        opt = Namespace(data=corpora, seed=-1)
        prefix_transform = prefix_cls(opt)
        prefix_transform.warm_up()
        self.assertIn("trainset", prefix_transform.prefix_dict)

        ex_in = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        with self.assertRaises(ValueError):
            prefix_transform.apply(ex_in)
            prefix_transform.apply(ex_in, corpus_name="validset")
        ex_out = prefix_transform.apply(ex_in, corpus_name="trainset")
        self.assertEqual(ex_out["src"][0], "｟_pf_src｠")
        self.assertEqual(ex_out["tgt"][0], "｟_pf_tgt｠")

    def test_filter_too_long(self):
        filter_cls = get_transforms_cls(["filtertoolong"])["filtertoolong"]
        opt = Namespace(src_seq_length=100, tgt_seq_length=100)
        filter_transform = filter_cls(opt)
        # filter_transform.warm_up()
        ex_in = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        ex_out = filter_transform.apply(ex_in)
        self.assertIs(ex_out, ex_in)
        filter_transform.tgt_seq_length = 2
        ex_out = filter_transform.apply(ex_in)
        self.assertIsNone(ex_out)


class TestSubwordTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_opts = {
            "seed": 3431,
            "share_vocab": False,
            "src_subword_model": "../../data/sample.bpe",
            "tgt_subword_model": "../../data/sample.bpe",
            "src_subword_nbest": 1,
            "tgt_subword_nbest": 1,
            "src_subword_alpha": 0.0,
            "tgt_subword_alpha": 0.0,
            "src_subword_vocab": "",
            "tgt_subword_vocab": "",
            "src_vocab_threshold": 0,
            "tgt_vocab_threshold": 0,
        }

    def test_sentencepiece(self):
        sp_cls = get_transforms_cls(["sentencepiece"])["sentencepiece"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_model"] = "../../data/sample.sp.model"
        base_opt["tgt_subword_model"] = "../../data/sample.sp.model"
        opt = Namespace(**base_opt)
        sp_cls._validate_options(opt)
        sp_transform = sp_cls(opt)
        sp_transform.warm_up()

        ex = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        sp_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ["▁H", "el", "lo", "▁world", "▁."],
            "tgt": ["▁B", "on", "j", "o", "ur", "▁le", "▁m", "on", "de", "▁."],
        }
        self.assertEqual(ex, ex_gold)

        # test SP regularization:
        sp_transform.src_subword_nbest = 4
        tokens = ["Another", "world", "."]
        gold_sp = ["▁An", "other", "▁world", "▁."]
        # 1. enable regularization for training example
        after_sp = sp_transform._tokenize(tokens, is_train=True)
        self.assertEqual(after_sp, ["▁An", "o", "ther", "▁world", "▁."])
        # 2. disable regularization for not training example
        after_sp = sp_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_sp, gold_sp)

        # Test mask location
        ex = {
            "src": "### Instruction: ｟newline｠instruction｟newline｠｟newline｠"
            "### Response : ｟newline｠｟_mask_before_｠response",
            "tgt": "",
        }
        ex["src"] = ex["src"].split(" ")
        ex_gold = {
            "src": [
                "▁",
                "#",
                "#",
                "#",
                "▁In",
                "struct",
                "ion",
                ":",
                "▁in",
                "struct",
                "ion",
                "▁",
                "#",
                "#",
                "#",
                "▁Re",
                "s",
                "p",
                "on",
                "s",
                "e",
                "▁",
                ":",
                "<blank>",
                "▁re",
                "s",
                "p",
                "on",
                "s",
                "e",
            ],
            "tgt": [],
        }
        sp_transform.apply(ex, is_train=True)
        self.assertEqual(ex, ex_gold)
