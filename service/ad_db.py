import random

from service.utils import get_logger


class ADDB:

    def get_ad(self, type):
        pass

    def log_ad(self, ad, where=None):
        pass


class ADList(ADDB):
    def __init__(self):
        self.all_ad = {"image": [("AD-20221020", "image", "./static/img/ad11.png", "https://www.hust.edu.cn"),
                                 ("AD-20221021", "image", "./static/img/ad1.png", "https://www.hust.edu.cn")],
                       "text": [("AD-20221022", "text", "广告内容1广告内容", "https://www.hust.edu.cn"),
                                ("AD-20221023", "text", "广告内容2广告内容", "https://www.hust.edu.cn")]
                       }

        self.logger_ad = get_logger(log_filename="ad.log", name="AD")

    def get_ad(self, type):
        ads = self.all_ad[type]
        n = len(ads)
        idx = random.randint(0, n - 1)

        return ads[idx]

    def log_ad(self, platform, ad, where=None):
        where = where+": " if where else "";
        self.logger_ad.info(platform + " " + where + str(ad))


ad_db = None


def get_addb():
    global ad_db
    if ad_db is not None:
        return ad_db

    ad_db = ADList()
    return ad_db
