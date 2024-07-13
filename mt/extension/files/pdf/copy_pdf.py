import fitz

from extension.files.pdf.copy_drawings import copy_drawings
from extension.files.pdf.copy_image import copy_images
from extension.files.pdf.copy_text import copy_text


if __name__ == "__main__":
    doc = fitz.open(r"D:/kidden/Conformer2020.pdf")
    outpdf = fitz.open()

    for page in doc:
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
        copy_drawings(page, outpage)
        copy_images(page, outpage, doc)
        copy_text(page, outpage)

    target_pdf_fn = "copy.pdf"
    outpdf.save(target_pdf_fn)
