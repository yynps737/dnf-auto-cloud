#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
服务器主模块，负责初始化和启动WebSocket服务
优化版 - 增加更好的资源管理和错误恢复
"""

import logging
import asyncio
import ssl
import sys
import os
import signal
import gc
import json
import time
import yaml
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.websocket_server import WebsocketServer
from config.settings import SECURITY, SERVER, MODEL
from models.yolo_model import YOLOModel
from utils.security import generate_ssl_cert
from utils.logging_utils import setup_logging

logger = logging.getLogger("DNFAutoCloud")

# 全局变量，用于优雅关闭
websocket_server = None
shutdown_event = asyncio.Event()

def setup_signals():
    """设置信号处理器"""
    def signal_handler(sig, frame):
        logger.info(f"收到信号 {sig}，准备关闭服务器...")
        shutdown_event.set()
    
    # 注册SIGINT和SIGTERM信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("信号处理器已设置")

async def cleanup():
    """清理资源"""
    logger.info("开始清理资源...")
    
    # 关闭WebSocket服务器
    global websocket_server
    if websocket_server:
        await websocket_server.stop()
    
    # 强制执行垃圾回收
    gc.collect()
    
    logger.info("资源清理完成")

async def startup_checks():
    """启动前检查"""
    logger.info("执行启动前检查...")
    
    # 检查模型权重
    if not os.path.exists(MODEL.get("weights", "")):
        logger.error(f"模型权重文件不存在: {MODEL.get('weights', '')}")
        return False
    
    # 检查GPU可用性
    if MODEL.get("device", "").startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("已配置使用CUDA，但无法找到可用的GPU。将使用CPU模式。")
                MODEL["device"] = "cpu"
            else:
                logger.info(f"已检测到GPU: {torch.cuda.get_device_name(0)}")
                # 打印CUDA版本等详细信息
                logger.info(f"CUDA版本: {torch.version.cuda}")
                logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        except ImportError:
            logger.warning("无法导入torch模块，将使用CPU模式。")
            MODEL["device"] = "cpu"
    
    # 检查SSL证书
    if SECURITY.get("ssl_enabled", False):
        if not os.path.exists(SECURITY.get("ssl_cert", "")) or not os.path.exists(SECURITY.get("ssl_key", "")):
            logger.info("未找到SSL证书，正在生成自签名证书...")
            try:
                generate_ssl_cert()
            except Exception as e:
                logger.error(f"生成SSL证书失败: {e}")
                return False
    
    # 检查数据目录
    data_dirs = ["data/logs", "data/cache", "data/sessions"]
    for dir_path in data_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logger.error(f"创建目录失败 {dir_path}: {e}")
            return False
    
    logger.info("启动前检查完成")
    return True

async def load_model():
    """加载YOLO模型"""
    logger.info("正在加载YOLO模型...")
    try:
        # 使用上下文管理器捕获并正确处理CUDA内存错误
        yolo_model = YOLOModel()
        logger.info("YOLO模型加载成功")
        
        # 打印模型性能基准
        performance = yolo_model.get_performance_stats()
        logger.info(f"模型性能基准: 平均推理时间 {performance['average_time']:.4f}秒 ({performance['fps']:.2f} FPS)")
        
        return yolo_model
    except Exception as e:
        logger.error(f"加载YOLO模型失败: {e}")
        return None

async def save_server_info(host, port, ssl_enabled):
    """保存服务器信息到文件，便于客户端连接"""
    info = {
        "server": {
            "host": host,
            "port": port,
            "url": f"{'wss' if ssl_enabled else 'ws'}://{host}:{port}/ws",
            "ssl_enabled": ssl_enabled
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.1"
    }
    
    # 保存为JSON
    try:
        with open("server_info.json", "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"服务器信息已保存到 server_info.json")
    except Exception as e:
        logger.error(f"保存服务器信息失败: {e}")

async def run_server(host, port, debug):
    """运行服务器"""
    global websocket_server
    
    # 执行启动前检查
    if not await startup_checks():
        logger.error("启动前检查失败，服务器未启动")
        return False
    
    # 加载YOLO模型
    yolo_model = await load_model()
    if not yolo_model:
        logger.error("加载模型失败，服务器未启动")
        return False
    
    # 创建SSL上下文（如果启用）
    ssl_context = None
    if SECURITY.get("ssl_enabled", False):
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(SECURITY.get("ssl_cert", ""), SECURITY.get("ssl_key", ""))
            logger.info("SSL证书加载成功")
        except Exception as e:
            logger.error(f"加载SSL证书失败: {e}")
            logger.warning("将以非SSL模式启动服务器")
    
    try:
        # 创建WebSocket服务器
        websocket_server = WebsocketServer(yolo_model, host, port, ssl_context)
        
        # 启动服务器
        await websocket_server.start()
        
        # 保存服务器信息
        await save_server_info(host, port, SECURITY.get("ssl_enabled", False))
        
        # 等待关闭信号
        await shutdown_event.wait()
        
        logger.info("收到关闭信号，正在停止服务器...")
        
        # 清理资源
        await cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"运行服务器时出错: {e}")
        return False

async def start_server_async(host="0.0.0.0", port=8080, debug=False):
    """异步启动服务器"""
    # 设置信号处理器
    setup_signals()
    
    # 运行服务器
    success = await run_server(host, port, debug)
    
    return success

def start_server(host="0.0.0.0", port=8080, debug=False):
    """同步启动服务器"""
    # 设置日志
    setup_logging(debug=debug)
    
    # 打印启动信息
    logger.info("=" * 50)
    logger.info("正在启动DNF自动化云服务...")
    logger.info(f"主机: {host}, 端口: {port}, 调试模式: {'开启' if debug else '关闭'}")
    logger.info(f"服务器版本: 1.0.1")
    logger.info("=" * 50)
    
    try:
        # 获取事件循环
        loop = asyncio.get_event_loop()
        if debug:
            loop.set_debug(True)
            logger.info("调试模式已启用")
        
        # 运行服务器
        success = loop.run_until_complete(start_server_async(host, port, debug))
        
        # 如果服务器启动成功，运行事件循环直到关闭
        if success:
            loop.run_until_complete(shutdown_event.wait())
        
        # 清理资源
        loop.run_until_complete(cleanup())
        
        # 关闭事件循环
        loop.close()
        
        logger.info("服务器已关闭")
        return success
        
    except KeyboardInterrupt:
        logger.info("接收到用户中断，正在关闭...")
        return False
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}")
        return False

if __name__ == "__main__":
    # 直接运行此脚本时的默认行为
    start_server(host="0.0.0.0", port=8080, debug=True)