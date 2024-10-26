"""Module that contain shard utils for dynamic data."""
from onmt.utils.logging import logger
from onmt.constants import CorpusName, CorpusTask
from onmt.transforms import TransformPipe
from contextlib import contextmanager
import itertools


@contextmanager
def exfile_open(filename, *args, **kwargs):
    """Extended file opener enables open(filename=None).

    This context manager enables open(filename=None) as well as regular file.
    filename None will produce endlessly None for each iterate,
    while filename with valid path will produce lines as usual.

    Args:
        filename (str|None): a valid file path or None;
        *args: args relate to open file using codecs;
        **kwargs: kwargs relate to open file using codecs.

    Yields:
        `None` repeatly if filename==None,
        else yield from file specified in `filename`.
    """
    if filename is None:
        from itertools import repeat

        _file = repeat(None)
    else:
        import codecs

        _file = codecs.open(filename, *args, **kwargs)
    yield _file
    if filename is not None and _file:
        _file.close()


class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(
        self, name, src, tgt, align=None):
        """Initialize src & tgt side file path."""
        self.id = name
        self.src = src
        self.tgt = tgt
        self.align = align

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        """

        def make_ex(sline, tline, align):
            # 'src_original' and 'tgt_original' store the
            # original line before tokenization. These
            # fields are used later on in the feature
            # transforms.
            example = {
                "src": sline,
                "tgt": tline,
                "src_original": sline,
                "tgt_original": tline,
            }
            if align is not None:
                example["align"] = align

            return example

        if isinstance(self.src, list):
            fs = self.src
            ft = [] if self.tgt is None else self.tgt
            fa = [] if self.align is None else self.align
            for i, (sline, tline, align) in enumerate(itertools.zip_longest(fs, ft, fa)):
                if (i // stride) % stride == offset:
                    yield make_ex(sline, tline, align)
        else:
            with exfile_open(self.src, mode="rb") as fs, exfile_open(
                self.tgt, mode="rb"
            ) as ft, exfile_open(self.align, mode="rb") as fa:
                for i, (sline, tline, align) in enumerate(zip(fs, ft, fa)):
                    if (i // stride) % stride == offset:
                        if tline is not None:
                            tline = tline.decode("utf-8")
                        if align is not None:
                            align = align.decode("utf-8")
                        yield make_ex(sline.decode("utf-8"), tline, align)

    def __str__(self):
        cls_name = type(self).__name__
        return (
            f"{cls_name}({self.id}, {self.src}, {self.tgt}, "
            f"align={self.align}, "
        )


def get_corpora(opts, task=CorpusTask.TRAIN, src=None, tgt=None, align=None):
    corpora_dict = {}
    if task == CorpusTask.TRAIN:  # 训练数据集
        for corpus_id, corpus_dict in opts.data.items():  # 多个训练数据集
            if corpus_id != CorpusName.VALID:  # 跳过data下的验证数据集
                corpora_dict[corpus_id] = ParallelCorpus(
                        corpus_id,
                        corpus_dict["path_src"],
                        corpus_dict["path_tgt"],
                    )
    elif task == CorpusTask.VALID:  # 验证数据集
        if CorpusName.VALID in opts.data.keys():
            corpora_dict[CorpusName.VALID] = ParallelCorpus(
                CorpusName.VALID,
                opts.data[CorpusName.VALID]["path_src"],
                opts.data[CorpusName.VALID]["path_tgt"] if tgt is None else None,
            )
        else:
            return None
    else:  # 推理数据集
        corpora_dict[CorpusName.INFER] = ParallelCorpus(
            CorpusName.INFER,
            src if src else opts.src,
            tgt if tgt else opts.tgt,
            align if align else None,
        )
    return corpora_dict


class ParallelCorpusIterator(object):
    """An iterator dedicated to ParallelCorpus.

    Args:
        corpus (ParallelCorpus): corpus to iterate;
        transform (TransformPipe): transforms to be applied to corpus;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate corpus with this line stride;
        offset (int): iterate corpus with this line offset.
    """

    def __init__(
        self, corpus, transform, skip_empty_level="warning", stride=1, offset=0
    ):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        if skip_empty_level not in ["silent", "warning", "error"]:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset

    def _process(self, stream):
        for i, example in enumerate(stream):
            example["src"] = example["src"].strip().split(" ")  # 按空格切分！！！
            example["src_original"] = example["src_original"].strip().split(" ")

            line_number = i * self.stride + self.offset
            example["cid_line_number"] = line_number
            example["cid"] = self.cid
            if "align" in example:
                example["align"] = example["align"].strip().split(" ")
            if example["tgt"] is not None:
                example["tgt"] = example["tgt"].strip().split(" ")
                example["tgt_original"] = example["tgt_original"].strip().split(" ")
                if (
                    len(example["src"]) == 0
                    or len(example["tgt"]) == 0
                    or ("align" in example and example["align"] == 0)
                ):
                    # empty example: skip
                    empty_msg = f"Empty line  in {self.cid}#{line_number}."
                    if self.skip_empty_level == "error":
                        raise IOError(empty_msg)
                    elif self.skip_empty_level == "warning":
                        logger.warning(empty_msg)
                    if len(example["src"]) == 0 and len(example["tgt"]) == 0:
                        yield (example, self.transform, self.cid)
                    continue
            yield (example, self.transform, self.cid)
        report_msg = self.transform.stats()
        if report_msg != "":
            logger.info(
                "* Transform statistics for {}({:.2f}%):\n{}\n".format(
                    self.cid, 100 / self.stride, report_msg
                )
            )

    def __iter__(self):
        """每个记录由字典类型的样本、转换和语料ID组成"""
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        corpus = self._process(corpus_stream)
        yield from corpus


def build_corpora_iters(
    corpora, transforms, corpora_info, skip_empty_level="warning", stride=1, offset=0
):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        transform_names = corpora_info[c_id].get("transforms", [])
        corpus_transform = [transforms[name] for name in transform_names if name in transforms]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        corpus_iter = ParallelCorpusIterator(
            corpus,
            transform_pipe,
            skip_empty_level=skip_empty_level,
            stride=stride,
            offset=offset,
        )
        corpora_iters[c_id] = corpus_iter
    return corpora_iters
