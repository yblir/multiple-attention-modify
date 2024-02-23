# -*- coding: utf-8 -*-
# @File: logger_util.py
# @Author: yblir
# @Time: 2022/12/11 下午 21:53
# @Explain: 设置logger日志输出格式
# ======================================================================================================================
import os
import sys

from loguru import logger


class MyLogger:
    def __init__(self, log_level="INFO", bool_std=True, bool_file=False, log_file_path=""):
        """
        :param bool_std: bool,为True时日志输出到控制台
        :param bool_file: bool,为True时日志输出到文件
        :param log_file_path: 待保存log日志的文件路径, 只有当bool_file为true时才有效
        :return:
        """
        if log_level not in ("DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"):
            logger.error(f'log_level must be in (DEBUG,INFO,SUCCESS,WARNING,ERROR,CRITICAL), but now is "{log_level}"')
            sys.exit()

        self.logger = logger
        # 清空所有设置
        self.logger.remove()

        # 分别为颜色>时间,日志等级,模块名 函数名 行号, 日志内容
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <red>|</red> " \
                     "<level>{level:<8}</level><red>|</red> " \
                     "<cyan>{module}.py</cyan>" \
                     ":<cyan>{line}</cyan> - " \
                     "<level>{message}</level>"

        # 设置控制台输出的格式,sys.bool_std为输出到屏幕
        if bool_std:
            self.logger.add(sys.stdout, level=log_level, format=log_format)

        if bool_file:
            # 日志文件路径校验
            if not isinstance(log_file_path, str):
                logger.error(f"log file path must be str, but now type is {type(log_file_path)}")
                sys.exit()
            elif not log_file_path.endswith('.log'):
                logger.error("log file path suffix must be log")
                sys.exit()
            elif not os.path.exists(log_file_path):
                # 判断是否为文件夹. 是就创建该文件夹. 不是就直接创建文件
                is_dir = "/" in log_file_path or "\\" in log_file_path
                is_dir and os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                handle_file = open(log_file_path, mode="w")
                handle_file.close()

            # 设置输出到文件的日志格式
            self.logger.add(log_file_path, level=log_level, format=log_format)

    def get_logger(self):
        return self.logger


if __name__ == '__main__':
    logger.info('test code')
    logger.debug('test code')
    logger.warning('test code')
    logger.error('test code')
    logger.success('test code')
    print('=======================================')

    my_logger = MyLogger(log_level="INFO", bool_std=True, bool_file=True, log_file_path='ab/test_log.log').get_logger()
    my_logger.info('test code')
    my_logger.debug('test code')
    my_logger.warning('test code')
    my_logger.error('test code')
    my_logger.success('test code')
