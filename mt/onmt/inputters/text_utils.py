import torch
from onmt.constants import DefaultTokens, CorpusTask, ModelTask
from torch.nn.utils.rnn import pad_sequence


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if ex["tgt"]:
        return len(ex["src"]["src_ids"]), len(ex["tgt"]["tgt_ids"])
    return len(ex["src"]["src_ids"])


def clean_example(maybe_example):
    maybe_example["src"] = {"src": " ".join(maybe_example["src"])}
    if maybe_example["tgt"] is not None:
        maybe_example["tgt"] = {"tgt": " ".join(maybe_example["tgt"])}

    return maybe_example


def process(task, bucket, **kwargs):
    """Returns valid transformed bucket from bucket."""
    transform_cid_to_examples = {}
    for example in bucket:
        transform_cid = (example[1], example[2])
        if transform_cid not in transform_cid_to_examples:
            transform_cid_to_examples[transform_cid] = []
        transform_cid_to_examples[transform_cid].append(example)

    processed_bucket = []
    # careful below it will return a bucket sorted by corpora
    # but we sort by length later and shuffle batches
    for (transform, cid), sub_bucket in transform_cid_to_examples.items():
        transf_bucket = transform.batch_apply(sub_bucket, is_train=(task == CorpusTask.TRAIN), corpus_name=cid)
        for example, transform, cid in transf_bucket:
            example = clean_example(example)  # 增加原始文本属性
            if len(example["src"]["src"]) > 0:
                processed_bucket.append(example)

        # at this point an example looks like:
        # {'src': {'src': ..., 'feats': [....]},
        #  'tgt': {'tgt': ...},
        #  'src_original': ['tok1', ...'tokn'],
        #  'tgt_original': ['tok1', ...'tokm'],
        #  'cid': corpus id
        #  'cid_line_number' : cid line number
        #  'align': ...,
        # }
    if len(processed_bucket) > 0:
        return processed_bucket
    else:
        return None


def numericalize(vocabs, example):
    """ """
    decoder_start_token = vocabs["decoder_start_token"]
    numeric = example
    numeric["src"]["src_ids"] = []
    src_text = example["src"]["src"].split(" ")  # XXX: 切分-合并-切分，多此一举
    numeric["src"]["src_ids"] = vocabs["src"](src_text)
    if example["tgt"] is not None:
        numeric["tgt"]["tgt_ids"] = []
        tgt_text = example["tgt"]["tgt"].split(" ")
        # 目标部分增加开始符和结束符
        numeric["tgt"]["tgt_ids"] = vocabs["tgt"]([decoder_start_token] + tgt_text + [DefaultTokens.EOS])

    return numeric


def tensorify(vocabs, minibatch, device, left_pad=False):
    """
    This function transforms a batch of example in tensors
    Each example looks like
    {'src': {'src': ..., 'feats': [...], 'src_ids': ...},
     'tgt': {'tgt': ..., 'tgt_ids': ...},
     'src_original': ['tok1', ...'tokn'],
     'tgt_original': ['tok1', ...'tokm'],
     'cid': corpus id
     'cid_line_number' : corpus id line number
     'ind_in_bucket': index in bucket
     'align': ...,
    }
    Returns  Dict of batch Tensors
        {'src': [seqlen, batchsize, n_feats+1],
         'tgt' : [seqlen, batchsize, n_feats=1],
         'cid': [batchsize],
         'cid_line_number' : [batchsize],
         'ind_in_bucket': [batchsize],
         'srclen': [batchsize],
         'tgtlen': [batchsize],
         'align': alignment sparse tensor
        }
    """
    tensor_batch = {}
    if left_pad:
        tbatchsrc = [
            torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device).flip(dims=[0])
            for ex, indice in minibatch
        ]
    else:
        tbatchsrc = [
            torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device)
            for ex, indice in minibatch
        ]

    # 填充
    padidx = vocabs["src"][DefaultTokens.PAD]
    tbatchsrc = pad_sequence(tbatchsrc, batch_first=True, padding_value=padidx)
    tbatchsrc = tbatchsrc[:, :, None]

    if left_pad:
        tensor_batch["src"] = tbatchsrc.flip(dims=[1])
    else:
        tensor_batch["src"] = tbatchsrc

    tensor_batch["srclen"] = torch.tensor(
        [len(ex["src"]["src_ids"]) for ex, indice in minibatch],
        dtype=torch.long,
        device=device,
    )

    if minibatch[0][0]["tgt"] is not None:
        if left_pad:
            tbatchtgt = [
                torch.tensor(ex["tgt"]["tgt_ids"], dtype=torch.long, device=device).flip(dims=[0])
                for ex, indice in minibatch
            ]
        else:
            tbatchtgt = [
                torch.tensor(ex["tgt"]["tgt_ids"], dtype=torch.long, device=device)
                for ex, indice in minibatch
            ]

        padidx = vocabs["tgt"][DefaultTokens.PAD]
        tbatchtgt = pad_sequence(tbatchtgt, batch_first=True, padding_value=padidx)
        tbatchtgt = tbatchtgt[:, :, None]
        tbatchtgtlen = torch.tensor(
            [len(ex["tgt"]["tgt_ids"]) for ex, indice in minibatch],
            dtype=torch.long,
            device=device,
        )
        if left_pad:
            tensor_batch["tgt"] = tbatchtgt.flip(dims=[1])
        else:
            tensor_batch["tgt"] = tbatchtgt
        tensor_batch["tgtlen"] = tbatchtgtlen

    tensor_batch["ind_in_bucket"] = [indice for ex, indice in minibatch]  # 桶内序号

    tensor_batch["cid"] = [ex["cid"] for ex, indice in minibatch]  # 语料编号
    tensor_batch["cid_line_number"] = [ex["cid_line_number"] for ex, indice in minibatch]  # 语料中序号

    return tensor_batch
