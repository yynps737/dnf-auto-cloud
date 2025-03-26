#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
服务器主模块，负责初始化和启动WebSocket服务
"""

import logging
import asyncio
import ssl
import sys
import os

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.websocket_server import WebsocketServer
from config.settings import SECURITY, SERVER
from models.yolo_model import YOLOModel
from utils.security import generate_ssl_cert

logger = logging.getLogger("DNFAutoCloud")

def start_server(host="0.0.0.0", port=8080, debug=False):
    """启动WebSocket服务器"""
    try:
        # 初始化YOLO模型
        logger.info("正在加载YOLO模型...")
        yolo_model = YOLOModel()
        
        # 创建SSL上下文（如果启用）
        ssl_context = None
        if SECURITY["ssl_enabled"]:
            if not os.path.exists(SECURITY["ssl_cert"]) or not os.path.exists(SECURITY["ssl_key"]):
                logger.info("未找到SSL证书，正在生成自签名证书...")
                generate_ssl_cert()
            
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(SECURITY["ssl_cert"], SECURITY["ssl_key"])
            logger.info("SSL证书加载成功")
        
        # 创建事件循环
        loop = asyncio.get_event_loop()
        if debug:
            loop.set_debug(True)
            logger.info("调试模式已启用")
        
        # 创建WebSocket服务器
        server = WebsocketServer(yolo_model, host, port, ssl_context)
        
        # 启动服务器
        logger.info(f"正在启动WebSocket服务器: {host}:{port}")
        loop.run_until_complete(server.start())
        
        # 运行服务器直到被中断
        logger.info("服务器启动成功，等待连接...")
        loop.run_forever()
        
    except KeyboardInterrupt:
        logger.info("服务器收到终止信号，正在关闭...")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        raise
    finally:
        # 清理资源
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        
        logger.info("正在关闭事件循环...")
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("服务器已关闭")