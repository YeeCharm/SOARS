# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------------

# import logging
# import os.path as osp

# from mmcv.utils import get_logger as get_root_logger
# from termcolor import colored

# logger_name = None


# def get_logger(cfg=None, log_level=logging.INFO):
#     global logger_name
#     if cfg is None:
#         return get_root_logger(logger_name)

#     # creating logger
#     name = cfg.model_name
#     output = cfg.output
#     logger_name = name

#     logger = get_root_logger(name, osp.join(output, "log.txt"), log_level=log_level, file_mode="a")
#     logger.propagate = False

#     fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
#     color_fmt = (
#         colored("[%(asctime)s %(name)s]", "green")
#         + colored("(%(filename)s %(lineno)d)", "yellow")
#         + ": %(levelname)s %(message)s"
#     )

#     for handler in logger.handlers:
#         if isinstance(handler, logging.StreamHandler):
#             handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))

#         if isinstance(handler, logging.FileHandler):
#             handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))

#     return logger





import logging
import os
import os.path as osp

from termcolor import colored

logger_name = None


def get_root_logger(
    name: str = None,
    log_file: str | None = None,
    log_level: int = logging.INFO,
    file_mode: str = "a",
) -> logging.Logger:
    """简易版 mmcv.utils.get_logger，仅用标准库 logging 实现。

    参数基本保持和 mmcv 一致：
    - name: logger 名（一般用模型名）
    - log_file: 日志文件路径（如 /path/to/output/log.txt）
    - log_level: 日志级别
    - file_mode: 文件打开模式，'a' 追加，'w' 覆盖
    """
    logger = logging.getLogger(name)

    # 如果已经有 handler，就不重复加 handler，避免重复打印
    if logger.handlers:
        logger.setLevel(log_level)
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    # 确保日志目录存在
    if log_file is not None:
        log_dir = osp.dirname(log_file)
        if log_dir and not osp.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # 先加一个简单的 StreamHandler，具体 formatter 在外层 get_logger 里设置
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    # 如果指定了文件，就再加一个 FileHandler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        logger.addHandler(file_handler)

    return logger


def get_logger(cfg=None, log_level=logging.INFO):
    """项目对外用的 logger 获取函数。

    - 如果 cfg 为 None：返回之前创建的同名 logger（用于在代码其它地方直接 get_logger()）
    - 如果 cfg 不为 None：使用 cfg.model_name 作为 logger 名，
      cfg.output/log.txt 作为日志文件路径。
    """
    global logger_name

    # 没有 cfg 的情况下，按原逻辑：用之前保存的 logger_name
    if cfg is None:
        return get_root_logger(logger_name)

    # 从 cfg 里取模型名和输出目录
    name = getattr(cfg, 'model_name', 'Talk2DINO')
    output = cfg.output
    logger_name = name

    # 创建/获取 logger
    logger = get_root_logger(
        name,
        osp.join(output, "log.txt"),
        log_level=log_level,
        file_mode="a",
    )
    logger.propagate = False

    # 日志格式（文件用普通格式，终端用彩色格式）
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(
                logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            )
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(
                logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
            )

    return logger

