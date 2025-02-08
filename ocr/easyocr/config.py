import os

import yaml

os.environ["LRU_CACHE_CAPACITY"] = "1"

# 读入检测器和识别器配置
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), encoding="utf-8") as stream:
    conf = yaml.safe_load(stream)

# 检测器配置
detection_models = conf["detection_models"]

# 识别器支持语言
latin_lang_list = ['af','az','bs','cs','cy','da','de','en','es','et','fr','ga',
                   'hr','hu','id','is','it','ku','la','lt','lv','mi','ms','mt',
                   'nl','no','oc','pi','pl','pt','ro','rs_latin','sk','sl','sq',
                   'sv','sw','tl','tr','uz','vi']
arabic_lang_list = ['ar','fa','ug','ur']
bengali_lang_list = ['bn','as','mni']
cyrillic_lang_list = ['ru','rs_cyrillic','be','bg','uk','mn','abq','ady','kbd',
                      'ava','dar','inh','che','lbe','lez','tab','tjk']
devanagari_lang_list = ['hi','mr','ne','bh','mai','ang','bho','mah','sck','new',
                        'gom','sa','bgc']
other_lang_list = ['th','ch_sim','ch_tra','ja','ko','ta','te','kn']

all_lang_list = latin_lang_list + arabic_lang_list+ cyrillic_lang_list +\
                devanagari_lang_list + bengali_lang_list + other_lang_list

# 识别器图片缺省高度
imgH = 64

# 识别器配置
recognition_models = conf["recognition_models"]
