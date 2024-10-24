import os
import random

from service.utils import get_logger

from collections import namedtuple

AD = namedtuple('AD', ['id', 'type', 'content', 'url'])


class ADDB:

    def __init__(self):
        # 广告展示记录日志
        self.logger_ad = get_logger(log_filename="ad.log", name="AD")

    def get_ad(self, type):
        """获得给定类型的广告"""
        ads = self.all_ad[type]
        n = len(ads)
        idx = random.randint(0, n - 1)

        return ads[idx]

    def log_ad(self, platform, ad_id, where=None, action="P"):
        """记录广告播放情况"""
        ad_msg = "|||".join([platform, where, ad_id, action])
        self.logger_ad.info(ad_msg)


class ADList(ADDB):
    """内存中广告DB测试类"""

    def __init__(self):
        super().__init__()

        self.all_ad = {"image": [("AD-20221020", "image", "./static/img/ad11.png", "https://www.hust.edu.cn"),
                                 ("AD-20221021", "image", "./static/img/ad1.png", "https://www.hust.edu.cn")],
                       "text": [("AD-20221022", "text", "广告内容1广告内容", "https://www.hust.edu.cn"),
                                ("AD-20221023", "text", "广告内容2广告内容", "https://www.hust.edu.cn")]
                       }

class ADFile(ADDB):

    def __init__(self, ad_file):
        super().__init__()

        import csv

        with open(ad_file, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            rows = list(reader)[1:]

        self.all_ad = {}

        for r in rows:
            ad = AD(*r)
            if ad.type not in self.all_ad:
                self.all_ad[ad.type] = []
            self.all_ad[ad.type].append(ad)

ad_db = None


def get_addb():
    global ad_db
    if ad_db is not None:
        return ad_db

    # ad_db = ADList()
    ad_db = ADFile(os.path.join(os.path.dirname(__file__), "ads.csv"))
    return ad_db


if __name__ == "__main__":
    ads = ADFile("./ads.csv")

    ad1 = ads.get_ad("text")
    print(ad1)
    print(ad1[0], ad1.id)
