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
    # print(blocks)
    for i, b in enumerate(blocks):  # iterate through the text blocks
        print("****block {}****".format(i+1))
        print(len(b["lines"]), "lines")
        print(b["number"], b["type"], b["bbox"])
        for j, l in enumerate(b["lines"]):  # iterate through the text lines
            print("*****line {}*****".format(j + 1))
            print(len(l["spans"]), "spans")
            for s in l["spans"]:  # iterate through the text spans
                print("")
                font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                    s["font"],  # font name
                    flags_decomposer(s["flags"]),  # readable font flags
                    s["size"],  # font size
                    s["color"],  # font color
                )
                print("Text: '%s'" % s["text"])  # simple print of text
                print(font_properties)


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

    # parse_page(doc[1])

    parse_shape(doc[0])


parse_pdf(r"D:\kidden\Conformer2020.pdf")


