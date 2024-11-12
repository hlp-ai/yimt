

font_dict = {
    "en": "helv",
    "zh": "china-ss",
}


def close_to(n1, n2, eps=3.0):
    return abs(n1-n2) < eps


def span_len(span):
    return len(span["text"].split())  # 空格分割的单词个数


def block_heigth(b):
    return b[3] - b[1]


def block_width(b):
    return b[2] - b[0]


def in_line(b1, b2, eps=3.0):
    # 两个bbox的y1和y2足够接近，则它们在一行内
    return close_to(b1[1], b2[1], eps) and close_to(b1[3], b2[3], eps)


def left_align(b1, b2, eps=16.0):
    return abs(b1[0]-b2[0]) < eps


def near_to(b1, b2, eps=9.9):
    # b1的x2足够靠近b2的x1
    return b2[0] - b1[2] < eps


def merge(block1, block2, sep_char=" "):
    bbox = (min(block1["bbox"][0], block2["bbox"][0]),
            min(block1["bbox"][1], block2["bbox"][1]),
            max(block1["bbox"][2], block2["bbox"][2]),
            max(block1["bbox"][3], block2["bbox"][3])
            )

    size = min(block1["size"], block2["size"])
    text = block1["text"] + sep_char + block2["text"]

    return {
        "bbox": bbox,
        "size": size,
        "text": text,
    }


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
             # "style": "",
             "size": min(sizes),
            #"dir": (1.0, 0.0)
             }]


def merge_spans(spans):
    if len(spans) == 1:
        return spans

    result = []
    while len(spans) > 1:
        block1 = spans[0]
        block2 = spans[1]
        if near_to(block1["bbox"], block2["bbox"]):  # 足够靠近，合并
            new_span = merge(block1, block2)
            spans.remove(block1)
            spans.remove(block2)
            spans.insert(0, new_span)  # 合并span取代原来两个span
        else:
            result.append(block1)  # 独立span
            spans.remove(block1)

    result.append(spans[0])

    return result


def merge_lines(lines):
    if len(lines) == 1:
        return lines

    result = []
    while len(lines) > 1:
        line1 = lines[0]
        if len(line1["spans"]) > 1:  # 只合并有1个span的行
            result.append(line1)
            lines.remove(line1)
            continue
        line2 = lines[1]
        if len(line2["spans"]) > 1:  # 只合并有1个span的行
            result.append(line2)
            lines.remove(line2)
            continue

        if in_line(line1["bbox"], line2["bbox"], 6.5) and near_to(line1["bbox"], line2["bbox"], 12.0):  # 在同一行，且足够靠近
            new_span = merge(line1["spans"][0], line2["spans"][0])
            lines.remove(line1)
            lines.remove(line2)
            new_line = {
                "bbox": new_span["bbox"],
                "spans": [new_span]
            }
            lines.insert(0, new_line)  # 合并line取代原来两个line
        else:
            result.append(line1)  # 独立行
            lines.remove(line1)

    result.append(lines[0])

    return result


def to_paragraph(block):
    blocks = []
    lines = block["lines"]

    if len(lines) == 1:
        for span in lines[0]["spans"]:
            blocks.append(span)
        return blocks
    else:
        while len(lines) > 1:
            line1 = lines[0]
            line2 = lines[1]
            if in_line(line1["bbox"], line2["bbox"], 6.5):  # 同一行未合并段
                blocks.append(line1["spans"][0])
                # for span in line1["spans"]:
                #    blocks.append(span)
                lines.remove(line1)
            elif left_align(line1["bbox"], line2["bbox"], 19.0):  # 不同行，左对齐，合并
                new_line = merge(line1["spans"][0], line2["spans"][0])
                # blocks.append(new_line)

                lines.remove(line1)
                lines.remove(line2)
                lines.insert(0, {
                    "bbox": new_line["bbox"],
                    "spans": [new_line]
                })
            else:  # 不同行，不对齐
                blocks.append(line1["spans"][0])
                lines.remove(line1)

    blocks.append(lines[0]["spans"][0])

    return blocks


def get_candidate_block(block):
    if "lines" not in block:
        return

    if len(block["lines"]) == 1:  # 单行块
        line = block["lines"][0]
        if len(line["spans"]) == 1:  # 单行单段
            span = line["spans"][0]
            return [{"text": span["text"],
                     "bbox": span["bbox"],
                     "size": span["size"]}]
        else:  # 单行多段
            return merge_spans(line["spans"])
            # lens = [span_len(s) for s in line["spans"]]
            # if sum(lens)/len(line["spans"]) < 3:  # 每段很短，各段独立
            #     return [{"text": s["text"],
            #              "bbox": s["bbox"],
            #              "size": s["size"]} for s in line["spans"]]
            # else:
            #     # 合并行内各段
            #     return merge_block(block)
    else:  # 多行块
        sizes = []
        lens = []
        for line in block["lines"]:
            for span in line["spans"]:
                sizes.append(span["size"])
                lens.append(span_len(span))

        if sum(lens) / len(lens) < 3:  # 每段很短，保留各行各段
            result = []
            for line in block["lines"]:
                result.extend([{"text": s["text"],
                                "bbox": s["bbox"],
                                "size": s["size"]} for s in line["spans"]])
            return result
        else:
            # 合并各行
            return merge_block(block)


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


# 简化页面块的标识
def simplify_page(page):
    blocks = page.get_text("dict")["blocks"]

    simple_page = []
    for block in blocks:
        lines = block["lines"]
        slines = []
        for line in lines:
            sspans = []
            for span in line["spans"]:
                sspan = {"bbox": simplify_floats(span["bbox"]),
                         "size": simplify_float(span["size"]),
                         "text": span["text"]}
                sspans.append(sspan)

            sline = {"spans": sspans,
                      "bbox": simplify_floats(line["bbox"])}
            slines.append(sline)

        sblock = {"lines": slines,
                  "bbox": simplify_floats(block["bbox"])}
        simple_page.append(sblock)

    return simple_page
