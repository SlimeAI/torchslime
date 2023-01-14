from ..util import NOTHING
import os

BASE_PATH = NOTHING
NAMESPACE = NOTHING
LOG_PATH = 'runtime.log'
METRIC_PATH = 'metrics.json'
CHECKPOINT_PATH = 'checkpoint'


def join_path(*args):
    return os.path.join(*args)


def set_base_path(path: str):
    global BASE_PATH
    BASE_PATH = path
    if os.path.exists(BASE_PATH) is False:
        os.makedirs(BASE_PATH)


def safe_makedirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def set_namespace(namespace: str):
    global NAMESPACE
    NAMESPACE = namespace
    namespace_path = join_path(BASE_PATH, NAMESPACE)
    if os.path.exists(namespace_path) is False:
        safe_makedirs(namespace_path)
    else:
        from . import logger
        logger.warn('The namespace folder already exists. Please check the namespace to avoid overwriting previous log files.')


def get_namespace_path():
    return join_path(BASE_PATH, NAMESPACE)


def get_log_path():
    return join_path(get_namespace_path(), LOG_PATH)


def get_metric_path():
    return join_path(get_namespace_path(), METRIC_PATH)


def get_checkpoint_path():
    return join_path(get_namespace_path(), CHECKPOINT_PATH)
