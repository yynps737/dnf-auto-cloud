#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DNF自动化云服务主启动脚本
"""

import os
import argparse
import logging
from server.main import start_server
from utils.logging_utils import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNF自动化云服务")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(debug=args.debug)
    logger = logging.getLogger("DNFAutoCloud")
    logger.info("正在启动DNF自动化云服务...")
    
    # 启动服务
    start_server(host=args.host, port=args.port, debug=args.debug)