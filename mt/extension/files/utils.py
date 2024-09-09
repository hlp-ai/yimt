import re
import threading


class Progress:

    def report(self, total, done, fid=None):
        pass


progress_lock = threading.Lock()


class TranslationProgress(Progress):
    def __init__(self):
        self.info_dict = {}

    def report(self, total, done, fid=None):
        if fid is None:
            return
        progress_info = "{}/{}".format(done, total)

        with progress_lock:
            self.info_dict[fid] = progress_info

        print("[Progress]", fid, progress_info)

    def set_info(self, info, fid):
        with progress_lock:
            self.info_dict[fid] = info

        print("[Progress]", fid, info)

    def get_info(self, fid=None):
        with progress_lock:
            for f, p in self.info_dict.items():
                if f == fid or f.endswith(fid):
                    return p

        return ""


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


coutry_codes = set([
    'AF',
    'AL',
    'DZ',
    'AS',
    'AD',
    'AO',
    'AI',
    'AG',
    'AR',
    'AM',
    'AW',
    'AU',
    'AT',
    'AZ',
    'BH',
    'BD',
    'BB',
    'BY',
    'BE',
    'BZ',
    'BJ',
    'BM',
    'BT',
    'BO',
    'BQ',
    'BA',
    'BW',
    'BR',
    'IO',
    'VG',
    'BN',
    'BG',
    'BF',
    'BI',
    'KH',
    'CM',
    'CA',
    'CV',
    'KY',
    'CI',
    'CF',
    'TD',
    'CL',
    'CN',
    'CO',
    'KM',
    'CK',
    'CR',
    'HR',
    'CW',
    'CY',
    'CZ',
    'CD',
    'DK',
    'DJ',
    'DM',
    'DO',
    'EC',
    'EG',
    'SV',
    'GQ',
    'ER',
    'EE',
    'ET',
    'FK',
    'FO',
    'FM',
    'FJ',
    'FI',
    'FR',
    'GF',
    'PF',
    'GA',
    'GE',
    'DE',
    'GH',
    'GI',
    'GR',
    'GL',
    'GD',
    'GP',
    'GU',
    'GT',
    'GG',
    'GN',
    'GW',
    'GY',
    'HT',
    'HN',
    'HK',
    'HU',
    'IS',
    'IN',
    'ID',
    'IQ',
    'IE',
    'IM',
    'IL',
    'IT',
    'JM',
    'JP',
    'JE',
    'JO',
    'KZ',
    'KE',
    'KI',
    'XK',
    'KW',
    'KG',
    'LA',
    'LV',
    'LB',
    'LS',
    'LR',
    'LY',
    'LI',
    'LT',
    'LU',
    'MO',
    'MK',
    'MG',
    'MW',
    'MY',
    'MV',
    'ML',
    'MT',
    'MH',
    'MQ',
    'MR',
    'MU',
    'YT',
    'MX',
    'MD',
    'MC',
    'MN',
    'ME',
    'MS',
    'MA',
    'MZ',
    'MM',
    'NA',
    'NR',
    'NP',
    'NL',
    'NC',
    'NZ',
    'NI',
    'NE',
    'NG',
    'NU',
    'NF',
    'MP',
    'NO',
    'OM',
    'PK',
    'PW',
    'PS',
    'PA',
    'PG',
    'PY',
    'PE',
    'PH',
    'PL',
    'PT',
    'PR',
    'QA',
    'RE',
    'CG',
    'RO',
    'RU',
    'RW',
    'BL',
    'SH',
    'KN',
    'MF',
    'PM',
    'VC',
    'WS',
    'SM',
    'ST',
    'SA',
    'SN',
    'RS',
    'SC',
    'SL',
    'SG',
    'SX',
    'SK',
    'SI',
    'SB',
    'SO',
    'ZA',
    'KR',
    'SS',
    'ES',
    'LK',
    'LC',
    'SD',
    'SR',
    'SZ',
    'SE',
    'CH',
    'TW',
    'TJ',
    'TZ',
    'TH',
    'BS',
    'GM',
    'TL',
    'TG',
    'TK',
    'TO',
    'TT',
    'TN',
    'TR',
    'TM',
    'TC',
    'TV',
    'UG',
    'UA',
    'AE',
    'GB',
    'US',
    'UY',
    'VI',
    'UZ',
    'VU',
    'VA',
    'VE',
    'VN',
    'WF',
    'EH',
    'YE',
    'ZM',
    'ZW',
])


def is_country_code(s):
    return s.upper() in coutry_codes;


lang_codes = set([
    'af',
    'af-za',
    'sq',
    'sq-al',
    'ar',
    'ar-dz',
    'ar-bh',
    'ar-eg',
    'ar-iq',
    'ar-jo',
    'ar-kw',
    'ar-lb',
    'ar-ly',
    'ar-ma',
    'ar-om',
    'ar-qa',
    'ar-sa',
    'ar-sy',
    'ar-tn',
    'ar-ae',
    'ar-ye',
    'hy',
    'hy-am',
    'az',
    'az-az-cyrl',
    'az-az-latn',
    'eu',
    'eu-es',
    'be',
    'be-by',
    'bg',
    'bg-bg',
    'ca',
    'ca-es',
    "zh",
    'zh-hk',
    'zh-mo',
    'zh-cn',
    'zh-chs',
    'zh-sg',
    'zh-tw',
    'zh-cht',
    'hr',
    'hr-hr',
    'cs',
    'cs-cz',
    'da',
    'da-dk',
    'div',
    'div-mv',
    'nl',
    'nl-be',
    'nl-nl',
    'en',
    'en-au',
    'en-bz',
    'en-ca',
    'en-cb',
    'en-ie',
    'en-jm',
    'en-nz',
    'en-ph',
    'en-za',
    'en-tt',
    'en-gb',
    'en-us',
    'en-zw',
    'et',
    'et-ee',
    'fo',
    'fo-fo',
    'fa',
    'fa-ir',
    'fi',
    'fi-fi',
    'fr',
    'fr-be',
    'fr-ca',
    'fr-fr',
    'fr-lu',
    'fr-mc',
    'fr-ch',
    'gl',
    'gl-es',
    'ka',
    'ka-ge',
    'de',
    'de-at',
    'de-de',
    'de-li',
    'de-lu',
    'de-ch',
    'el',
    'el-gr',
    'gu',
    'gu-in',
    'he',
    'he-il',
    'hi',
    'hi-in',
    'hu',
    'hu-hu',
    'is',
    'is-is',
    'id',
    'id-id',
    'it',
    'it-it',
    'it-ch',
    'ja',
    'ja-jp',
    'kn',
    'kn-in',
    'kk',
    'kk-kz',
    'kok',
    'kok-in',
    'ko',
    'ko-kr',
    'ky',
    'ky-kz',
    'lv',
    'lv-lv',
    'lt',
    'lt-lt',
    'mk',
    'mk-mk',
    'ms',
    'ms-bn',
    'ms-my',
    'mr',
    'mr-in',
    'mn',
    'mn-mn',
    'no',
    'nb-no',
    'nn-no',
    'pl',
    'pl-pl',
    'pt',
    'pt-br',
    'pt-pt',
    'pa',
    'pa-in',
    'ro',
    'ro-ro',
    'ru',
    'ru-ru',
    'sa',
    'sa-in',
    'sr-sp-cyrl',
    'sr-sp-latn',
    'sk',
    'sk-sk',
    'sl',
    'sl-si',
    'es',
    'es-ar',
    'es-bo',
    'es-cl',
    'es-co',
    'es-cr',
    'es-do',
    'es-ec',
    'es-sv',
    'es-gt',
    'es-hn',
    'es-mx',
    'es-ni',
    'es-pa',
    'es-py',
    'es-pe',
    'es-pr',
    'es-es',
    'es-uy',
    'es-ve',
    'sw',
    'sw-ke',
    'sv',
    'sv-fi',
    'sv-se',
    'syr',
    'syr-sy',
    'ta',
    'ta-in',
    'tt',
    'tt-ru',
    'te',
    'te-in',
    'th',
    'th-th',
    'tr',
    'tr-tr',
    'uk',
    'uk-ua',
    'ur',
    'ur-pk',
    'uz',
    'uz-uz-cyrl',
    'uz-uz-latn',
    'vi',
    'vi-vn',
])


def is_lang_code(s):
    return s.lower() in lang_codes


en_letter = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
en_letter_set = set([c for c in en_letter])


def has_no_en_letter(s):
    for c in s:
        if c in en_letter_set:
            return False

    return True


def should_translate(txt, lang="en"):
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

    if is_country_code(txt):
        return False

    if is_lang_code(txt):
        return False

    if lang == "en" and has_no_en_letter(txt):
        return False

    return True


def doc2docx(doc_fn, docx_fn=None, keep_active=True):
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


def ppt2pptx(ppt_fn, pptx_fn=None, keep_active=True):
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


def xls2xlsx(xls_fn, xlsx_fn, keep_active=True):
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
    print(should_translate("en"))
    print(should_translate("ZH"))
    print(should_translate("cc"))

    doc2docx(r"d:\kidden\test.doc")
    ppt2pptx(r"d:\kidden\enzh1.ppt")
