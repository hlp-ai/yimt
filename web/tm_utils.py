import re


PATTERN = re.compile(r"<lang-pair>(.+?)</lang-pair>\n<source>(.+?)</source>\n<target>(.+?)</target>",
                         re.MULTILINE | re.DOTALL)

class TMList:

    def __init__(self, file):
        self.records = []
        with open(file, encoding="utf-8") as f:
            content = f.read()

        for m in re.finditer(PATTERN, content):
            self.records.append({
                "direction": m.group(1).strip(),
                "source": m.group(2).strip(),
                "target": m.group(3).strip()
            })





