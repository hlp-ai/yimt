import argparse

import pymupdf


def get_block_text(block):
    text = ""
    for line in block["lines"]:
        line_text = ""
        for span in line["spans"]:
            line_text += span["text"]

        text += line_text + "\n"

    return text.replace("-\n", "").replace("\n", " ")



def get_page_text(page):
    blocks = page.get_text("dict")["blocks"]
    # pprint(blocks)

    text = ""
    for i, b in enumerate(blocks):
        if b["type"] != 0:
            continue

        text += get_block_text(b) + "\n\n"

    return text


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-i", "--input", type=str, required=True, help="输入PDF文件路径")
    arg_parser.add_argument("-o", "--output", type=str, default=None, help="结果P文件路径")
    args = arg_parser.parse_args()

    in_fn = args.input

    doc = pymupdf.open(in_fn)

    text = ""
    for page in doc:
        text += get_page_text(page)

    if args.output is None:
        out_fn = in_fn.replace(".pdf", ".txt")
    else:
        out_fn = args.output

    with open(out_fn, "w", encoding="utf-8") as f:
        f.write(text)
