#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WebSocket服务器实现，处理客户端连接和通信 - Linux兼容版
"""

import asyncio
import json
import logging
import time
import uuid
import base64
from datetime import datetime
from io import BytesIO

import websockets
from PIL import Image

from config.settings import SERVER, SECURITY
from server.image_processor import process_image
from server.action_generator import generate_actions
from utils.security import decrypt_message, encrypt_message
from utils.human_behavior import generate_human_delay

logger = logging.getLogger("DNFAutoCloud")

class WebsocketServer:
    """WebSocket服务器类 - Linux兼容版"""
    
    def __init__(self, yolo_model, host="0.0.0.0", port=8080, ssl_context=None):
        """初始化WebSocket服务器"""
        self.yolo_model = yolo_model
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.clients = {}  # 客户端连接管理
        self.start_time = datetime.now()
    
    async def start(self):
        """启动WebSocket服务器"""
        return await websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            max_size=10 * 1024 * 1024,  # 10MB最大消息大小
            ping_interval=SERVER["heartbeat_interval"],
            ping_timeout=SERVER["timeout"]
        )
    
    async def handle_connection(self, websocket, path):
        """处理新的WebSocket连接"""
        client_id = str(uuid.uuid4())
        client_info = {
            "id": client_id,
            "connected_at": datetime.now(),
            "remote": websocket.remote_address,
            "last_activity": time.time()
        }
        
        # 限制最大连接数
        if len(self.clients) >= SERVER["max_connections"]:
            logger.warning(f"达到最大连接数限制，拒绝新连接: {client_info['remote']}")
            await websocket.close(1013, "服务器连接数已满")
            return
        
        # 添加到客户端列表
        self.clients[client_id] = client_info
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
        finally:
            # 清理客户端连接
            if client_id in self.clients:
                del self.clients[client_id]
            logger.info(f"客户端连接已关闭: {client_id}")
    
    async def _authenticate(self, websocket, client_id):
        """客户端认证"""
        try:
            # 发送认证挑战
            challenge = str(uuid.uuid4())
            await websocket.send(json.dumps({
                "type": "auth_challenge",
                "challenge": challenge
            }))
            
            # 等待认证响应
            response_raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=SERVER["timeout"]
            )
            
            # 检查认证响应
            response = json.loads(response_raw)
            if response.get("type") != "auth_response":
                logger.warning(f"客户端认证格式错误: {client_id}")
                await websocket.close(1008, "认证格式错误")
                return False
            
            # 这里可以实现更复杂的认证逻辑
            # 简单示例仅检查是否有响应
            if "response" in response:
                # 认证成功
                self.clients[client_id]["authenticated"] = True
                await websocket.send(json.dumps({
                    "type": "auth_result",
                    "status": "success",
                    "client_id": client_id
                }))
                logger.info(f"客户端认证成功: {client_id}")
                return True
            else:
                # 认证失败
                await websocket.close(1008, "认证失败")
                logger.warning(f"客户端认证失败: {client_id}")
                return False
                
        except asyncio.TimeoutError:
            logger.warning(f"客户端认证超时: {client_id}")
            await websocket.close(1008, "认证超时")
            return False
        except Exception as e:
            logger.error(f"客户端认证出错: {client_id}, 错误: {e}")
            await websocket.close(1011, "认证处理错误")
            return False
    
    async def _handle_messages(self, websocket, client_id):
        """处理客户端消息"""
        while True:
            # 接收消息
            message_raw = await websocket.recv()
            
            # 更新最后活动时间
            self.clients[client_id]["last_activity"] = time.time()
            
            try:
                # 解析消息（可能需要解密）
                if SECURITY["encryption_enabled"]:
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
                    await self._handle_heartbeat(websocket, client_id)
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
            except Exception as e:
                logger.error(f"处理消息时出错，来自客户端: {client_id}, 错误: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "processing_error",
                    "message": f"处理消息时出错: {str(e)}"
                }))
    
    async def _handle_image_request(self, websocket, client_id, message_data):
        """处理图像识别请求"""
        try:
            # 提取图像数据
            image_base64 = message_data.get("data", "")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            
            # 记录请求
            logger.debug(f"收到图像识别请求，来自客户端: {client_id}, 图像尺寸: {image.size}")
            
            # 处理图像并进行检测
            detections = process_image(self.yolo_model, image)
            
            # 根据检测结果生成动作
            game_state = message_data.get("game_state", {})
            actions = generate_actions(detections, game_state)
            
            # 生成人类化延迟
            delay = generate_human_delay()
            
            # 构建响应
            response = {
                "type": "action_response",
                "request_id": message_data.get("request_id"),
                "timestamp": time.time(),
                "actions": actions,
                "delay": delay
            }
            
            # 发送响应（可能需要加密）
            if SECURITY["encryption_enabled"]:
                await websocket.send(encrypt_message(response))
            else:
                await websocket.send(json.dumps(response))
            
            logger.debug(f"已发送动作响应，客户端: {client_id}, 动作数: {len(actions)}")
            
        except Exception as e:
            logger.error(f"处理图像请求时出错，客户端: {client_id}, 错误: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": "image_processing_error",
                "request_id": message_data.get("request_id"),
                "message": f"处理图像时出错: {str(e)}"
            }))
    
    async def _handle_heartbeat(self, websocket, client_id):
        """处理心跳消息"""
        try:
            # 构建心跳响应
            response = {
                "type": "heartbeat_response",
                "timestamp": time.time(),
                "server_uptime": (datetime.now() - self.start_time).total_seconds()
            }
            
            # 发送响应
            await websocket.send(json.dumps(response))
            logger.debug(f"已发送心跳响应，客户端: {client_id}")
            
        except Exception as e:
            logger.error(f"处理心跳消息时出错，客户端: {client_id}, 错误: {e}")