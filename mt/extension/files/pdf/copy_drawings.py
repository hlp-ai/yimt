import pymupdf


def copy_drawings(page, outpage):
    paths = page.get_drawings()  # extract existing drawings

    # this is a list of "paths", which can directly be drawn again using Shape
    # -------------------------------------------------------------------------
    #
    # define some output page with the same dimensions
    # outpdf = pymupdf.open()
    # outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
    shape = outpage.new_shape()  # make a drawing canvas for the output page

    # --------------------------------------
    # loop through the paths and draw them
    # --------------------------------------
    for path in paths:
        # ------------------------------------
        # draw each entry of the 'items' list
        # ------------------------------------
        for item in path["items"]:  # these are the draw commands
            if item[0] == "l":  # line
                shape.draw_line(item[1], item[2])
            elif item[0] == "re":  # rectangle
                shape.draw_rect(item[1])
            elif item[0] == "qu":  # quad
                shape.draw_quad(item[1])
            elif item[0] == "c":  # curve
                shape.draw_bezier(item[1], item[2], item[3], item[4])
            else:
                raise ValueError("unhandled drawing", item)

        # ------------------------------------------------------
        # all items are drawn, now apply the common properties
        # to finish the path
        # ------------------------------------------------------
        lineCap = 0 if path["lineCap"] is None else max(path["lineCap"])
        lineJoin = 0 if path["lineJoin"] is None else path["lineJoin"]
        stroke_opacity = 1 if path["stroke_opacity"] is None else path["stroke_opacity"]
        fill_opacity = 1 if path["fill_opacity"] is None else path["fill_opacity"]
        shape.finish(
            fill=path["fill"],  # fill color
            color=path["color"],  # line color
            dashes=path["dashes"],  # line dashing
            even_odd=path.get("even_odd", True),  # control color of overlaps
            closePath=path["closePath"],  # whether to connect last and first point
            lineJoin=lineJoin,  # how line joins should look like
            lineCap=lineCap,  # how line ends should look like
            width=path["width"],  # line width
            stroke_opacity=stroke_opacity,  # same value for both
            fill_opacity=fill_opacity,  # opacity parameters
        )

    # all paths processed - commit the shape to its page
    shape.commit()


if __name__ == "__main__":
    doc = pymupdf.open(r"D:/kidden/GKBM.pdf")
    outpdf = pymupdf.open()
    for page in doc:
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
        copy_drawings(page, outpage)

    outpdf.save("copy-drawings.pdf")