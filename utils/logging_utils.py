#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具，提供日志记录功能
"""

import os
import logging
import logging.handlers
import time
from datetime import datetime

from config.settings import LOGGING, BASE_DIR

def setup_logging(debug=False):
    """
    设置日志系统
    
    参数:
        debug (bool): 是否启用调试模式
    """
    # 创建日志目录
    log_dir = os.path.dirname(LOGGING["file"])
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 设置日志级别
    log_level = logging.DEBUG if debug else getattr(logging, LOGGING["level"], logging.INFO)
    root_logger.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        LOGGING["file"],
        maxBytes=LOGGING["max_size"],
        backupCount=LOGGING["backup_count"],
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 应用格式化器
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 记录启动信息
    logger = logging.getLogger("DNFAutoCloud")
    logger.info(f"日志系统初始化完成，级别: {logging.getLevelName(log_level)}")
    logger.info(f"日志文件: {LOGGING['file']}")
    
    if debug:
        logger.info("调试模式已启用")