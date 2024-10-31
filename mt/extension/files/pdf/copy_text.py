import sys
from pprint import pprint

import pymupdf


def copy_text(page, outpage):
    shape = outpage.new_shape()
    # read page text as a dictionary, suppressing extra spaces in CJK fonts
    blocks = page.get_text("dict")["blocks"]
    # print(len(blocks), "blocks")
    # pprint(blocks)
    for i, b in enumerate(blocks):  # iterate through the text blocks
        # print("****block {}****".format(i+1))
        if b["type"] != 0:  # XXX:为什么这里有图片？
            continue

        pprint(b)

        #print(len(b["lines"]), "lines")
        #print(b["number"], b["type"], b["bbox"])
        for j, l in enumerate(b["lines"]):  # iterate through the text lines
            cos, sin = l["dir"]
            rotate = 0
            if cos == 0 and sin == -1:  # TODO: 方向处理全面
                rotate = 90
            #print("*****line {}*****".format(j + 1))
            #print(len(l["spans"]), "spans")
            for s in l["spans"]:  # iterate through the text spans
                # print("color", s["color"])
                # TODO: 颜色处理，字体处理，文本方向处理，文本区域大小
                shape.insert_textbox(s["bbox"], s["text"],
                                     fontsize=s["size"]*0.85,
                                     rotate=rotate,
                                     lineheight=0.05,)

                # shape.insert_text(s["origin"], s["text"],
                #                   fontsize=s["size"] - 0.8,
                #                   rotate=rotate)

                # shape.insert_text(s["origin"], s["text"],
                #                   fontsize=s["size"], fontname="china-ss")

                # print("")
                # font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                #     s["font"],  # font name
                #     flags_decomposer(s["flags"]),  # readable font flags
                #     s["size"],  # font size
                #     s["color"],  # font color
                # )
                # print("Text: '%s'" % s["text"])  # simple print of text
                # print(font_properties)

    shape.commit()


if __name__ == "__main__":
    doc = pymupdf.open(sys.argv[1])
    outpdf = pymupdf.open()
    for page in doc:
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
        copy_text(page, outpage)

    outpdf.save("copy-text.pdf")