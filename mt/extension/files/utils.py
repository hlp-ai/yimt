import re


def is_number(s):
    try:
        float(s)  # 尝试将字符串转换为浮点数
        return True
    except ValueError:
        return False


def is_empty(s):
    s = s.strip()
    if len(s) == 0:
        return True

    return False


def is_special_symbol(s):
    return s in ["th", "rd", "nd", "st"]


url_pattern = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", re.IGNORECASE)
email_pattern = re.compile(r'^mailto:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', re.IGNORECASE)


def is_url(url):
    if url_pattern.match(url):
        return True

    if email_pattern.match(url):
        return True

    return False


def should_translate(txt):
    if is_empty(txt):
        return False

    txt = txt.strip()

    if len(txt) == 1:
        return False

    if is_special_symbol(txt):
        return False

    if is_number(txt):
        return False

    if is_url(txt):
        return False

    return True


def doc2docx(doc_fn, docx_fn=None, keep_active = True):
    if docx_fn is None:
        docx_fn = doc_fn[:-4] + ".docx"

    from win32com import client

    word_app = client.Dispatch("Word.Application")
    doc = word_app.Documents.Open(doc_fn)
    try:
        doc.SaveAs2(docx_fn, FileFormat=16)
    except:
        raise
    finally:
        doc.Close(0)

    if not keep_active:
        word_app.Quit()


def ppt2pptx(ppt_fn, pptx_fn=None, keep_active = True):
    if pptx_fn is None:
        pptx_fn = ppt_fn[:-4] + ".pptx"

    from win32com import client

    ppt_app = client.Dispatch("Powerpoint.Application")
    presentation = ppt_app.Presentations.Open(ppt_fn, 0, 0, 0)
    try:
        presentation.SaveAs(pptx_fn, FileFormat=24)
    except:
        raise
    finally:
        presentation.Close()

    if not keep_active:
        ppt_app.Quit()


def xls2xlsx(xls_fn, xlsx_fn, keep_active = True):
    if xlsx_fn is None:
        xls_fn = xls_fn[:-4] + ".xlsx"

    from win32com import client

    excel_app = client.Dispatch("Excel.Application")
    sheet = excel_app.Workbooks.Open(xls_fn)
    try:
        sheet.SaveAs(xlsx_fn, FileFormat=51)
    except:
        raise
    finally:
        sheet.Close(0)

    if not keep_active:
        excel_app.Quit()


if __name__ == "__main__":
    print(should_translate("http://www.abc.com"))
    print(should_translate("https://www.abc.com?q=v"))
    print(should_translate("mailto:abc@abc.com"))
    print(should_translate("DDD DD"))
    print(should_translate("ABCCC"))
    print(should_translate("  "))
    print(should_translate("12"))
    print(should_translate("12.45"))
    print(should_translate("-22.45"))

    doc2docx(r"d:\kidden\test.doc")
    ppt2pptx(r"d:\kidden\enzh1.ppt")