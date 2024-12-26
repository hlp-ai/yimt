import re


PATTERN = re.compile(r"<lang-pair>(.+?)</lang-pair>\n<source>(.+?)</source>\n<target>(.+?)</target>",
                         re.MULTILINE | re.DOTALL)

class TMList:

    def __init__(self, file):
        self.records = []
        self.tm_f = file
        with open(file, encoding="utf-8") as f:
            content = f.read()

        for m in re.finditer(PATTERN, content):
            self.records.append({
                "direction": m.group(1).strip(),
                "source": m.group(2).strip(),
                "target": m.group(3).strip()
            })

    def save(self, tm_f=None):
        if tm_f is None:
            tm_f = self.tm_f

        with open(tm_f, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(
                    f"<lang-pair>{r['direction']}</lang-pair>\n<source>{r['source']}</source>\n<target>{r['target']}</target>\n\n")





