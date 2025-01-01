

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
    # 两行左边接近，并且第二行右边不超过第一行或者两行右边接近
    return abs(b1[0]-b2[0]) < eps and (b1[2]>b2[2] or abs(b1[2]-b2[2])<3.0)


def near_to(b1, b2, eps=9.9):
    # b1的x2足够靠近b2的x1
    return b2[0] - b1[2] < eps


def merge(block1, block2, sep_char=" "):
    """block结构为: bbox, size, text"""
    # 两个bbox的最大外包bbox
    bbox = (min(block1["bbox"][0], block2["bbox"][0]),
            min(block1["bbox"][1], block2["bbox"][1]),
            max(block1["bbox"][2], block2["bbox"][2]),
            max(block1["bbox"][3], block2["bbox"][3])
            )

    # 字体大小去小的
    size = min(block1["size"], block2["size"])

    # 合并文本
    text = block1["text"] + sep_char + block2["text"]

    return {
        "bbox": bbox,
        "size": size,
        "text": text,
    }


def merge_spans(spans):
    if len(spans) == 1:
        return spans

    result = []
    while len(spans) > 1:
        block1 = spans[0]
        block2 = spans[1]
        if near_to(block1["bbox"], block2["bbox"]):  # 足够靠近，合并
            new_span = merge(block1, block2, sep_char="")
            spans.remove(block1)
            spans.remove(block2)
            spans.insert(0, new_span)  # 合并span取代原来两个span
        else:
            result.append(block1)  # 独立span
            spans.remove(block1)

    result.append(spans[0])

    return result


def long_enough(t1, t2, threshold=2):
    n1 = len(t1.split())
    n2 = len(t2.split())
    return (n1+n2)/2 > threshold


def all_char(t1, t2):
    for d in '0123456789':
        if d in t1 or d in t2:
            return False

    return True


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

        if in_line(line1["bbox"], line2["bbox"], 7.5) and near_to(line1["bbox"], line2["bbox"], 13.0) and (
                long_enough(line1["spans"][0]["text"], line2["spans"][0]["text"])
                or all_char(line1["spans"][0]["text"], line2["spans"][0]["text"])):  # 在同一行，且足够靠近，且段包含足够数量单词
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
        if block["type"] != 0:  # XXX:为什么这里有图片？
            continue

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


def blocks_for_translation(page):
    # print("===简化页面===")
    blocks = simplify_page(page)
    # pprint(blocks)

    for block in blocks:
        for line in block["lines"]:
            spans = line["spans"]
            new_spans = merge_spans(spans)
            line["spans"] = new_spans

    # print("===合并SPAN页面===")
    # pprint(blocks)

    for block in blocks:
        lines = block["lines"]
        new_lines = merge_lines(lines)
        block["lines"] = new_lines

    # print("===合并LINE页面===")
    # pprint(blocks)

    result = []

    # print("===翻译段落===")
    for block in blocks:
        paragraphs = to_paragraph(block)
        #pprint(paragraphs)
        result.extend(paragraphs)

    return result


if __name__ == "__main__":
    pass
