import datetime
import os
import logging
from src.settings import settings


root_logger = logging.getLogger()
root_logger.setLevel(settings.LOG_LEVEL)


def get_file_handler():
    today = str(datetime.date.today())
    dir_path = os.path.join(settings.LOG_FOLDER, today)
    log_path = os.path.join(dir_path, settings.LOG_FILE)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(settings.LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    return file_handler


def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(settings.LOG_LEVEL)
    stream_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    return stream_handler


def init_logger():
    root_logger.setLevel(settings.LOG_LEVEL)
    root_logger.addHandler(get_file_handler())
    root_logger.addHandler(get_stream_handler())
    
    if not os.path.exists(settings.LOG_FOLDER):
        os.makedirs(settings.LOG_FOLDER) 

    return root_logger


def change_log_file():
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.addHandler(get_file_handler())


init_logger()
logger = logging.getLogger(__name__)