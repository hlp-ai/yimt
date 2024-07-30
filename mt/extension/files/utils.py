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

    if is_number(txt):
        return False

    if is_url(txt):
        return False

    return True


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