import random


class ADDB:

    def get_ad(self, type):
        pass


class ADList(ADDB):
    def __init__(self):
        self.all_ad = {"image": [("AD-20221020", "image", "./static/img/ad11.png", "url-a"),
                                 ("AD-20221021", "image", "./static/img/ad1.png", "url-b")],
                       "text": [("AD-20221022", "type", "广告内容1广告内容", "url-c"),
                                ("AD-20221023", "type", "广告内容2广告内容", "url-d")]
                       }

    def get_ad(self, type):
        ads = self.all_ad[type]
        n = len(ads)
        idx = random.randint(0, n - 1)
        return ads[idx]


ad_db = None


def get_addb():
    global ad_db
    if ad_db is not None:
        return ad_db

    ad_db = ADList()
    return ad_db