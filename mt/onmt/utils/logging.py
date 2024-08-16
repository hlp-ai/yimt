# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler

# placing this here make it easier to call logger.info
# from anywhere, just 'from onmt.utils.logging import logger'
logger = logging.getLogger("onmt")


# 获得控制台和文件日志记录器
def init_logger(
    log_file=None,
    log_file_level=logging.NOTSET,  # 缺省父日志级别
    rotate=False,
    log_level=logging.INFO,
):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger("onmt")
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != "":
        if rotate:
            file_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
