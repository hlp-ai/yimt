import logging
from functools import wraps
import os
from logging.handlers import RotatingFileHandler

import langid


class NoLimiter:
    def exempt(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper


class SuspiciousFileOperation(Exception):
    pass


def path_traversal_check(unsafe_path, known_safe_path):
    known_safe_path = os.path.abspath(known_safe_path)
    unsafe_path = os.path.abspath(unsafe_path)

    if (os.path.commonprefix([known_safe_path, unsafe_path]) != known_safe_path):
        raise SuspiciousFileOperation("{} is not safe".format(unsafe_path))

    # Passes the check
    return unsafe_path


def get_logger(log_filename, name="mt"):
    """Get a logger with RotatingFileHandler and StreamHandler with DEBUG level"""
    logger = logging.getLogger(name)
    # fn = "{}.log".format(log_filename)
    formatter = logging.Formatter("%(asctime)s - %(name)s-%(levelname)s %(message)s")

    max_bytes = 64*1024*1024
    fh = RotatingFileHandler(log_filename, maxBytes=max_bytes, encoding="utf-8", backupCount=8)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.setLevel(logging.DEBUG)

    return logger


def detect_lang(text):
    """Detect language of text

    Args:
        text: text to be detected

    Returns:
        language code string
    """
    # text = remove_lang_independent(text)
    if all([c.isascii() for c in text]):
        return "en"
    return langid.classify(text)[0]


def is_valid_url(url):
    from urllib.parse import urlparse

    try:
        result = urlparse(url)
        return True
    except ValueError:
        return False


def get_page(url):
    import requests

    r = requests.get(url)

    if r.status_code != 200:
        return None

    if not r.headers["Content-Type"].startswith("text"):
        return None

    return r.content.decode(r.encoding)
