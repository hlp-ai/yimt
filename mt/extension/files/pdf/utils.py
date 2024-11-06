

font_dict = {
    "en": "helv",
    "zh": "china-ss",
}


def span_len(span):
    return len(span["text"].split())  # 空格分割的单词个数


def merge_block(block):
    sizes = []  # 各段字体大小
    for line in block["lines"]:
        for span in line["spans"]:
            sizes.append(span["size"])

    text = ""
    for line in block["lines"]:
        line_text = ""
        for span in line["spans"]:
            line_text += span["text"] + " "
        text += line_text + "\n"  # 换行

    return [{"text": text,
             "bbox": block["bbox"],
             "style": "",
             "size": min(sizes),  # 取最小字体
             "dir": (1.0, 0.0)}]


def get_candidate_block(block):
    if "lines" not in block:
        return

    return merge_block(block)

    # if len(block["lines"]) == 1:  # 单行块
    #     line = block["lines"][0]
    #     if len(line["spans"]) == 1:  # 单行单段
    #         span = line["spans"][0]
    #         return [{"text": span["text"],
    #                  "bbox": span["bbox"],
    #                  "style": flags_decomposer(span["flags"]),
    #                  "size": span["size"],
    #                  "dir": line["dir"]}]
    #     else:  # 单行多段
    #         lens = [span_len(s) for s in line["spans"]]
    #         if sum(lens)/len(line["spans"]) < 3:  # 每段很短，各段独立
    #             return [{"text": s["text"],
    #                      "bbox": s["bbox"],
    #                      "style": flags_decomposer(s["flags"]),
    #                      "size": s["size"],
    #                      "dir": line["dir"]} for s in line["spans"]]
    #         else:
    #             # 合并行内各段
    #             return merge_block(block)
    # else:  # 多行块
    #     fonts = []
    #     sizes = []
    #     lens = []
    #     for line in block["lines"]:
    #         for span in line["spans"]:
    #             fonts.append(span["font"])
    #             sizes.append(span["size"])
    #             lens.append(span_len(span))
    #
    #     if sum(lens) / len(lens) < 3:  # 每段很短，保留各行各段
    #         result = []
    #         for line in block["lines"]:
    #             result.extend([{"text": s["text"],
    #                             "bbox": s["bbox"],
    #                             "style": flags_decomposer(s["flags"]),
    #                             "size": s["size"],
    #                             "dir": (1.0, 0.0)} for s in line["spans"]])
    #         return result
    #     else:
    #         # 合并各行
    #         return merge_block(block)


def flags_decomposer(flags):
    """可读字体标志"""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)


def simplify_float(n):
    f = float("{:.2f}".format(n))
    return f


def simplify_floats(ns):
    return [simplify_float(n) for n in ns]
