# -*- coding: utf-8 -*-

import torch
from itertools import accumulate
from onmt.constants import SubwordMarker


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


# Helper functions
def begin_uppercase(token):
    return token == SubwordMarker.BEGIN_UPPERCASE


def end_uppercase(token):
    return token == SubwordMarker.END_UPPERCASE


def begin_case(token):
    return token == SubwordMarker.BEGIN_CASED


def case_markup(token):
    return begin_uppercase(token) or end_uppercase(token) or begin_case(token)


def subword_map_by_joiner(
    subwords, original_subwords=None, marker=SubwordMarker.JOINER
):
    """Return word id for each subword token (annotate by joiner)."""

    flags = [1] * len(subwords)
    j = 0
    finished = True
    for i, tok in enumerate(subwords):
        previous_tok = subwords[i - 1] if i else ""  # Previous N-1 token
        previous_tok_2 = subwords[i - 2] if i > 1 else ""  # Previous N-2 token
        # Keeps track of the original words/subwords
        # ('prior_tokenization' option)
        current_original_subword = (
            ""
            if not original_subwords
            else original_subwords[j]
            if j < len(original_subwords)
            else ""
        )

        if tok.startswith(marker) and tok != current_original_subword:
            flags[i] = 0
        elif (
            previous_tok.endswith(marker)
            or begin_case(previous_tok)
            or begin_uppercase(previous_tok)
        ) and not finished:
            flags[i] = 0
        elif (
            previous_tok_2.endswith(marker)
            and case_markup(previous_tok)
            and not finished
        ):
            flags[i] = 0
        elif end_uppercase(tok) and tok != current_original_subword:
            flags[i] = 0
        else:
            finished = False
            if tok == current_original_subword:
                finished = True
            j += 1

    flags[0] = 0
    word_group = list(accumulate(flags))

    if original_subwords:
        assert max(word_group) < len(original_subwords)
    return word_group


def subword_map_by_spacer(subwords, marker=SubwordMarker.SPACER):
    """Return word id for each subword token (annotate by spacer)."""
    flags = [0] * len(subwords)
    for i, tok in enumerate(subwords):
        if marker in tok:
            if case_markup(tok.replace(marker, "")):
                if i < len(subwords) - 1:
                    flags[i] = 1
            else:
                if i > 0:
                    previous = subwords[i - 1].replace(marker, "")
                    if not case_markup(previous):
                        flags[i] = 1

    # In case there is a final case_markup when new_spacer is on
    for i in range(1, len(subwords) - 1):
        if case_markup(subwords[-i]):
            flags[-i] = 0
        elif subwords[-i] == marker:
            flags[-i] = 0
            break

    word_group = list(accumulate(flags))
    if word_group[0] == 1:  # when dummy prefix is set
        word_group = [item - 1 for item in word_group]
    return word_group
