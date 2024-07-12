import fitz

from extension.files.copy_image import copy_images

doc = fitz.open(r"D:/kidden/GKBM.pdf")
outpdf = fitz.open()

for page in doc:
    outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
    copy_images(page, outpage, doc)


target_pdf_fn = "copy-img.pdf"
outpdf.save(target_pdf_fn)