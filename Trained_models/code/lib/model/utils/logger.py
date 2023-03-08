# coding:utf-8
import os
import sys
import io
import datetime
import logging
import traceback


def create_detail_day():
    daytime = datetime.datetime.now().strftime('day_' + '%Y_%m_%d')
    detail_time = daytime
    return detail_time


def create_logger(path, file_name):
    logging.basicConfig()
    logger = logging.getLogger()
    # 这个setLevel的作用是控制输出到控制台的信息
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = os.path.join(path, file_name)
    file_handler = logging.FileHandler(log_file)
    # file_handler的设置级别应该是对于输出的 .log文件的
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def make_print_to_file(path, filename):
    class Logger(object):
        def __init__(self, level="info", ):
            self.level = level

        def write(self, message):
            if message != '\n' and message != ' ' \
                    and message != '\t' and message != '    ' and message != '':
                if message[-1] == '\n':
                    message = message[:-1]
                if self.level == 'info':
                    # 这里是指所有的log信息
                    logger.info(message)
                elif self.level == 'error':
                    logger.error(message)

        def flush(self):
            pass

    folder = os.path.join(path, filename.split('.')[0])
    if not os.path.exists(folder):
        os.makedirs(folder)

    logger = create_logger(folder, create_detail_day() + '.log')
    sys.stdout = Logger(level='info')
    sys.stderr = Logger(level='error')
