from math import floor
from pprint import pprint

import pymupdf


def flags_decomposer(flags):
    """Make font flags human readable."""
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


def dump_span(span):
    text = span["text"]
    size = simplify_float(span["size"])
    font = span["font"]
    bbox = simplify_floats(span["bbox"])
    origin = simplify_floats(span["origin"])
    flags = flags_decomposer(span["flags"])
    color = span["color"]
    ascender = simplify_float(span["ascender"])
    descender = simplify_float(span["descender"])

    return f"\t\t\t<span bbox='{bbox}' origin='{origin}' size='{size}' color='{color}' font='{font}' style='{flags}' ascender='{ascender}' descender='{descender}'>{text}</span>"


def dump_line(line):
    bbox = simplify_floats(line["bbox"])
    dir = line["dir"]

    spans = [dump_span(s) for s in line["spans"]]
    spans = "\n".join(spans)

    return f"\t\t<line bbox='{bbox}' dir='{dir}'>\n{spans}\n\t\t</line>"


def dump_block(block):
    bbox = simplify_floats(block["bbox"])
    type = block["type"]
    number = block["number"]

    lines = [dump_line(line) for line in block["lines"]]
    lines = "\n".join(lines)

    return f"\t<block bbox='{bbox}' type='{type}' number='{number}'>\n{lines}\n\t</block>"


def dump_page(page):
    blocks = page.get_text("dict")["blocks"]
    blocks = [dump_block(block) for block in blocks]
    blocks = "\n".join(blocks)

    mediabox = (simplify_float(page.mediabox.x0), simplify_float(page.mediabox.y0),
                simplify_float(page.mediabox.x1), simplify_float(page.mediabox.y1))

    rect = (simplify_float(page.rect.x0), simplify_float(page.rect.y0),
                simplify_float(page.rect.x1), simplify_float(page.rect.y1))

    return f"<page mediabox='{mediabox}' rect='{rect}' rotation='{page.rotation}' number='{page.number}'>\n{blocks}\n</page>"


def parse_page(page):
    print(page.mediabox)
    print(page.rect)
    print(page.rotation)
    print(page.xref)
    print(page.number)
    print(page.parent)
    print()

    # pprint(page.get_links())
    #
    # for a in page.annots():
    #     pprint(a)

    # pprint(page.get_text("blocks"))
    # for block in page.get_text("blocks"):
    #     print(block)
    #     print()

    # read page text as a dictionary, suppressing extra spaces in CJK fonts
    blocks = page.get_text("dict")["blocks"]
    print(len(blocks), "blocks")
    # pprint(blocks)
    for i, b in enumerate(blocks):  # iterate through the text blocks
        print("****block {}****".format(i+1))
        pprint(b)
        print(len(b["lines"]), "lines")
        # print(b["number"], b["type"], b["bbox"])
        for j, l in enumerate(b["lines"]):  # iterate through the text lines
            print("*****line {}*****".format(j + 1))
            print(len(l["spans"]), "spans")
            for s in l["spans"]:  # iterate through the text spans
                font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                    s["font"],  # font name
                    flags_decomposer(s["flags"]),  # readable font flags
                    s["size"],  # font size
                    s["color"],  # font color
                )
                print("Text: '%s'" % s["text"])  # simple print of text
                print(font_properties)
            print()

        print("\n")


def parse_shape(page):
    drawings = page.get_drawings()
    print("{} drawings".format(len(drawings)))
    for i, d in enumerate(drawings):
        print("***drawing{}***".format(i+1))
        pprint(d)


def parse_pdf(fn):
    doc = pymupdf.open(fn)
    pprint(doc.metadata)
    print()

    print(doc.language)
    print(doc.outline)
    print(doc.page_count)
    print(doc.name)
    print(doc.pagemode)
    print(doc.pagelayout)
    print(doc.is_encrypted)
    print(doc.chapter_count)
    print(doc.is_reflowable)
    print(doc.FormFonts)
    print(doc.FontInfos)

    print()

    # parse_page(doc[0])
    print(dump_page(doc[0]))

    # parse_shape(doc[0])


parse_pdf(r"D:\kidden\Conformer2020.pdf")


