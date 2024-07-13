import os
import sys

import fitz
import pymupdf

from extension.files.pdf.copy_drawings import copy_drawings
from extension.files.pdf.copy_image import copy_images
from extension.files.pdf.copy_text import copy_text


if __name__ == "__main__":
    pdf_file = sys.argv[1]
    copy_fn = os.path.join(os.path.dirname(pdf_file), "copy-" + os.path.basename(pdf_file))

    doc = pymupdf.open(pdf_file)
    outpdf = fitz.open()

    for page in doc:
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
        copy_drawings(page, outpage)
        copy_images(page, outpage, doc)
        copy_text(page, outpage)

    outpdf.save(copy_fn)
