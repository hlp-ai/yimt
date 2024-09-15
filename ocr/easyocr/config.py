import os

import yaml

os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.environ.get("EASYOCR_MODULE_PATH") or \
              os.environ.get("MODULE_PATH") or \
              os.path.expanduser("~/.EasyOCR/")


with open(os.path.join(os.path.dirname(__file__), "config.yaml"), encoding="utf-8") as stream:
    conf = yaml.safe_load(stream)

detection_models = conf["detection_models"]

# recognizer parameters
latin_lang_list = ['af','az','bs','cs','cy','da','de','en','es','et','fr','ga',\
                   'hr','hu','id','is','it','ku','la','lt','lv','mi','ms','mt',\
                   'nl','no','oc','pi','pl','pt','ro','rs_latin','sk','sl','sq',\
                   'sv','sw','tl','tr','uz','vi']
arabic_lang_list = ['ar','fa','ug','ur']
bengali_lang_list = ['bn','as','mni']
cyrillic_lang_list = ['ru','rs_cyrillic','be','bg','uk','mn','abq','ady','kbd',\
                      'ava','dar','inh','che','lbe','lez','tab','tjk']
devanagari_lang_list = ['hi','mr','ne','bh','mai','ang','bho','mah','sck','new',\
                        'gom','sa','bgc']
other_lang_list = ['th','ch_sim','ch_tra','ja','ko','ta','te','kn']

all_lang_list = latin_lang_list + arabic_lang_list+ cyrillic_lang_list +\
                devanagari_lang_list + bengali_lang_list + other_lang_list
imgH = 64
# separator_list = {
#     'th': ['\xa2', '\xa3'],
#     'en': ['\xa4', '\xa5']
# }
# separator_char = []
# for lang, sep in separator_list.items():
#     separator_char += sep

recognition_models = conf["recognition_models"]
