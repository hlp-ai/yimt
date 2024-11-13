import torch


def build_align_pharaoh(valid_alignment):
    """Convert valid alignment matrix to i-j (from 0) Pharaoh format pairs,
    or empty list if it's None.
    """
    align_pairs = []
    align_scores = []
    if isinstance(valid_alignment, torch.Tensor):
        tgt_align_src_id = valid_alignment.argmax(dim=-1)
        align_scores = torch.divide(
            valid_alignment.max(dim=-1).values, valid_alignment.sum(dim=-1)
        )
        for tgt_id, src_id in enumerate(tgt_align_src_id.tolist()):
            align_pairs.append(str(src_id) + "-" + str(tgt_id))
        align_scores = [
            "{0}-{1:.5f}".format(i, s) for i, s in enumerate(align_scores.tolist())
        ]
        align_pairs.sort(key=lambda x: int(x.split("-")[-1]))  # sort by tgt_id
        align_pairs.sort(key=lambda x: int(x.split("-")[0]))  # sort by src_id
        print(align_scores)
    return align_pairs, align_scores

