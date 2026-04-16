"""统一日志配置。

所有模块通过 `get_logger(__name__)` 获取 logger，
不使用裸 print()。
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """获取模块级 logger。

    Args:
        name:  通常传入 ``__name__``。
        level: 日志级别，默认继承 root logger 配置。

    Returns:
        配置好 handler 的 Logger 实例。
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    # 禁止向 root logger 传播，避免 configure_root_logger 配置后日志重复输出
    logger.propagate = False

    if level is not None:
        logger.setLevel(level)

    return logger


def configure_root_logger(level: int = logging.INFO) -> None:
    """配置 root logger，应在程序入口（main.py）调用一次。

    同时更新所有已创建的子 logger 级别（它们设置了 propagate=False，
    不会自动继承 root 的级别，因此必须在此处显式同步）。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 同步更新所有已初始化的子 logger 级别（--debug 标志生效的关键）
    for logger_obj in logging.Logger.manager.loggerDict.values():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.setLevel(level)
