import os
import tempfile
import fitz


imgdir = tempfile.mkdtemp()


def recoverpix(doc, item):
    xref = item[0]  # xref of PDF image
    smask = item[1]  # xref of its /SMask

    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        if pix0.n > 3:
            ext = "pam"
        else:
            ext = "png"

        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {  # create dictionary expected by caller
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png"),
        }

    return doc.extract_image(xref)


def copy_images(page, outpage, in_pdf):
    images = page.get_images()
    # print(len(images), "images")
    # pprint(images)
    for img in images:
        xref = img[0]
        img_rect = page.get_image_rects(xref)
        # print(img_rect)

        image = recoverpix(in_pdf, img)
        imgdata = image["image"]

        imgfile = os.path.join(imgdir, "img%05i.%s" % (xref, image["ext"]))
        # print(imgfile)
        with open(imgfile, "wb") as fout:
            fout.write(imgdata)

        if len(img_rect) > 0:
            outpage.insert_image(rect=img_rect[0], stream=imgdata)


if __name__ == "__main__":
    doc = fitz.open(r"D:/kidden/vits2021.pdf")
    outpdf = fitz.open()

    for page in doc:
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
        copy_images(page, outpage, doc)

    target_pdf_fn = "copy-img.pdf"
    outpdf.save(target_pdf_fn)
