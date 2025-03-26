#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WebSocket服务器实现，处理客户端连接和通信 - 优化版本
增强稳定性、连接管理和会话持久化
"""

import asyncio
import json
import logging
import time
import uuid
import base64
import os
from datetime import datetime
from io import BytesIO
import threading

import websockets
from PIL import Image
import numpy as np

from config.settings import SERVER, SECURITY
from server.image_processor import process_image
from server.action_generator import generate_actions
from utils.security import decrypt_message, encrypt_message
from utils.human_behavior import generate_human_delay

logger = logging.getLogger("DNFAutoCloud")

class WebsocketServer:
    """WebSocket服务器类 - 优化版本"""
    
    def __init__(self, yolo_model, host="0.0.0.0", port=8080, ssl_context=None):
        """初始化WebSocket服务器"""
        self.yolo_model = yolo_model
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.clients = {}  # 客户端连接管理
        self.client_sessions = {}  # 客户端会话数据
        self.start_time = datetime.now()
        self.shutdown_event = asyncio.Event()
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "images_processed": 0,
            "actions_generated": 0,
            "errors": 0
        }
        
        # 客户端监控线程
        self.monitoring_task = None
        
        # 保存最近处理的图像(用于调试)
        self.debug_mode = os.environ.get("DNF_DEBUG", "0") == "1"
        if self.debug_mode:
            os.makedirs("debug/images", exist_ok=True)
    
    async def start(self):
        """启动WebSocket服务器"""
        ws_server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            max_size=10 * 1024 * 1024,  # 10MB最大消息大小
            ping_interval=SERVER.get("heartbeat_interval", 5),
            ping_timeout=SERVER.get("timeout", 60),
            close_timeout=10
        )
        
        # 启动客户端监控
        self.monitoring_task = asyncio.create_task(self.monitor_clients())
        
        logger.info(f"WebSocket服务器已启动: {self.host}:{self.port}")
        return ws_server
    
    async def stop(self):
        """停止WebSocket服务器"""
        logger.info("正在停止WebSocket服务器...")
        
        # 设置关闭事件
        self.shutdown_event.set()
        
        # 关闭所有客户端连接
        close_tasks = []
        for client_id, info in list(self.clients.items()):
            if "websocket" in info:
                try:
                    close_tasks.append(info["websocket"].close(1001, "服务器关闭"))
                except:
                    pass
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # 停止监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("WebSocket服务器已停止")
    
    async def monitor_clients(self):
        """监控客户端连接状态"""
        inactive_timeout = SERVER.get("inactive_timeout", 300)  # 默认5分钟
        
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # 检查不活跃的客户端
                for client_id, info in list(self.clients.items()):
                    if "last_activity" in info:
                        inactive_time = current_time - info["last_activity"]
                        
                        # 清理超时客户端
                        if inactive_time > inactive_timeout:
                            logger.info(f"客户端 {client_id} 不活跃超过 {inactive_timeout} 秒，断开连接")
                            try:
                                if "websocket" in info:
                                    await info["websocket"].close(1000, "不活跃超时")
                            except:
                                pass
                            
                            if client_id in self.clients:
                                del self.clients[client_id]
                                self.stats["active_connections"] -= 1
                
                # 更新服务器状态
                self.stats["active_connections"] = len(self.clients)
                
                # 等待下一个检查周期
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控客户端出错: {e}")
                await asyncio.sleep(60)  # 出错后等待一分钟再重试
    
    async def handle_connection(self, websocket, path=None):
        """处理新的WebSocket连接"""
        client_id = str(uuid.uuid4())
        
        # 创建客户端信息
        client_info = {
            "id": client_id,
            "connected_at": datetime.now(),
            "remote": websocket.remote_address,
            "last_activity": time.time(),
            "websocket": websocket,
            "authenticated": False
        }
        
        # 限制最大连接数
        if len(self.clients) >= SERVER.get("max_connections", 10):
            logger.warning(f"达到最大连接数限制，拒绝新连接: {client_info['remote']}")
            await websocket.close(1013, "服务器连接数已满")
            return
        
        # 添加到客户端列表
        self.clients[client_id] = client_info
        self.stats["total_connections"] += 1
        self.stats["active_connections"] += 1
        
        logger.info(f"新客户端连接: {client_id} 来自 {client_info['remote']}")
        
        try:
            # 客户端认证
            authenticated = await self._authenticate(websocket, client_id)
            if not authenticated:
                logger.warning(f"客户端认证失败: {client_id}")
                return
            
            # 处理消息
            await self._handle_messages(websocket, client_id)
            
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"客户端连接关闭: {client_id}, 代码: {e.code}, 原因: {e.reason}")
        except Exception as e:
            logger.error(f"处理客户端连接时出错: {client_id}, 错误: {e}")
            self.stats["errors"] += 1
        finally:
            # 清理客户端连接
            if client_id in self.clients:
                del self.clients[client_id]
                self.stats["active_connections"] -= 1
            logger.info(f"客户端连接已关闭: {client_id}")
    
    async def _authenticate(self, websocket, client_id):
        """客户端认证"""
        try:
            # 发送认证挑战
            challenge = str(uuid.uuid4())
            await websocket.send(json.dumps({
                "type": "auth_challenge",
                "challenge": challenge,
                "timestamp": time.time()
            }))
            
            # 等待认证响应
            response_raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=SERVER.get("timeout", 60)
            )
            
            # 检查认证响应
            response = json.loads(response_raw)
            if response.get("type") != "auth_response":
                logger.warning(f"客户端认证格式错误: {client_id}")
                await websocket.close(1008, "认证格式错误")
                return False
            
            # 获取客户端信息
            client_info = response.get("client_info", {})
            
            # 保存会话信息
            self.client_sessions[client_id] = {
                "client_info": client_info,
                "game_state": {},
                "last_actions": [],
                "statistics": {
                    "images_processed": 0,
                    "actions_sent": 0,
                    "errors": 0
                }
            }
            
            # 认证成功
            self.clients[client_id]["authenticated"] = True
            self.clients[client_id]["client_info"] = client_info
            
            await websocket.send(json.dumps({
                "type": "auth_result",
                "status": "success",
                "client_id": client_id,
                "server_info": {
                    "version": "1.0.1",
                    "uptime": (datetime.now() - self.start_time).total_seconds()
                }
            }))
            
            logger.info(f"客户端认证成功: {client_id}, 版本: {client_info.get('version', 'unknown')}")
            return True
                
        except asyncio.TimeoutError:
            logger.warning(f"客户端认证超时: {client_id}")
            await websocket.close(1008, "认证超时")
            return False
        except Exception as e:
            logger.error(f"客户端认证出错: {client_id}, 错误: {e}")
            await websocket.close(1011, "认证处理错误")
            self.stats["errors"] += 1
            return False
    
    async def _handle_messages(self, websocket, client_id):
        """处理客户端消息"""
        while not self.shutdown_event.is_set():
            # 接收消息
            message_raw = await websocket.recv()
            
            # 更新最后活动时间
            self.clients[client_id]["last_activity"] = time.time()
            
            try:
                # 解析消息（可能需要解密）
                if SECURITY.get("encryption_enabled", False):
                    message_data = decrypt_message(message_raw)
                else:
                    message_data = json.loads(message_raw)
                
                # 处理不同类型的消息
                msg_type = message_data.get("type", "")
                
                if msg_type == "image":
                    # 处理图像识别请求
                    await self._handle_image_request(websocket, client_id, message_data)
                elif msg_type == "heartbeat":
                    # 处理心跳消息
                    await self._handle_heartbeat(websocket, client_id, message_data)
                else:
                    # 未知消息类型
                    logger.warning(f"收到未知消息类型: {msg_type} 来自客户端: {client_id}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "unknown_message_type",
                        "message": f"未知消息类型: {msg_type}"
                    }))
            
            except json.JSONDecodeError:
                logger.error(f"JSON解析错误，来自客户端: {client_id}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "invalid_json",
                    "message": "无效的JSON格式"
                }))
                self.stats["errors"] += 1
            except Exception as e:
                logger.error(f"处理消息时出错，来自客户端: {client_id}, 错误: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "processing_error",
                    "message": f"处理消息时出错: {str(e)}"
                }))
                self.stats["errors"] += 1
    
    async def _handle_image_request(self, websocket, client_id, message_data):
        """处理图像识别请求"""
        try:
            start_time = time.time()
            
            # 提取图像数据
            image_base64 = message_data.get("data", "")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            
            # 提取游戏状态
            game_state = message_data.get("game_state", {})
            window_rect = message_data.get("window_rect", [0, 0, 0, 0])
            
            # 更新会话中的游戏状态
            if client_id in self.client_sessions:
                self.client_sessions[client_id]["game_state"] = game_state
            
            # 保存调试图像
            if self.debug_mode:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = f"debug/images/{client_id}_{timestamp}.jpg"
                try:
                    image.save(debug_path)
                except:
                    pass
            
            # 处理图像并进行检测
            detections = process_image(self.yolo_model, image)
            
            # 根据检测结果生成动作
            actions = generate_actions(detections, game_state)
            
            # 生成人类化延迟
            delay = generate_human_delay()
            
            # 追踪处理时间
            processing_time = time.time() - start_time
            
            # 构建响应
            response = {
                "type": "action_response",
                "request_id": message_data.get("request_id"),
                "timestamp": time.time(),
                "actions": actions,
                "detections": detections,  # 返回检测结果供客户端分析
                "delay": delay,
                "processing_time": processing_time
            }
            
            # 发送响应（可能需要加密）
            if SECURITY.get("encryption_enabled", False):
                await websocket.send(encrypt_message(response))
            else:
                await websocket.send(json.dumps(response))
            
            # 更新统计信息
            self.stats["images_processed"] += 1
            self.stats["actions_generated"] += len(actions)
            
            if client_id in self.client_sessions:
                self.client_sessions[client_id]["statistics"]["images_processed"] += 1
                self.client_sessions[client_id]["statistics"]["actions_sent"] += len(actions)
                self.client_sessions[client_id]["last_actions"] = actions
            
            # 记录处理信息
            if len(actions) > 0:
                logger.debug(f"已发送 {len(actions)} 个动作给客户端: {client_id}, 处理时间: {processing_time:.3f}秒")
            
        except Exception as e:
            logger.error(f"处理图像请求时出错，客户端: {client_id}, 错误: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": "image_processing_error",
                "request_id": message_data.get("request_id"),
                "message": f"处理图像时出错: {str(e)}"
            }))
            self.stats["errors"] += 1
            if client_id in self.client_sessions:
                self.client_sessions[client_id]["statistics"]["errors"] += 1
    
    async def _handle_heartbeat(self, websocket, client_id, message_data):
        """处理心跳消息"""
        try:
            # 提取客户端游戏状态
            if "game_state" in message_data and client_id in self.client_sessions:
                self.client_sessions[client_id]["game_state"].update(message_data["game_state"])
            
            # 构建心跳响应
            response = {
                "type": "heartbeat_response",
                "timestamp": time.time(),
                "server_uptime": (datetime.now() - self.start_time).total_seconds(),
                "server_stats": {
                    "connections": len(self.clients),
                    "images_processed": self.stats["images_processed"],
                    "actions_generated": self.stats["actions_generated"]
                }
            }
            
            # 发送响应
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"处理心跳消息时出错，客户端: {client_id}, 错误: {e}")
            self.stats["errors"] += 1
    
    def get_server_stats(self):
        """获取服务器统计信息"""
        return {
            "start_time": self.start_time.isoformat(),
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "connections": {
                "total": self.stats["total_connections"],
                "active": self.stats["active_connections"]
            },
            "performance": {
                "images_processed": self.stats["images_processed"],
                "actions_generated": self.stats["actions_generated"],
                "errors": self.stats["errors"]
            },
            "clients": [
                {
                    "id": client_id,
                    "connected_at": info["connected_at"].isoformat(),
                    "remote": info["remote"],
                    "authenticated": info["authenticated"],
                    "version": info.get("client_info", {}).get("version", "unknown")
                }
                for client_id, info in self.clients.items()
            ]
        }