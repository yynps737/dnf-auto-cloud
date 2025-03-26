#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DNF自动化客户端，负责截取游戏画面并执行服务器返回的操作
"""

import os
import sys
import json
import base64
import time
import random
import asyncio
import websockets
import configparser
import ssl
import ctypes
from io import BytesIO
from datetime import datetime
import threading
import logging
import traceback

# 图像处理
from PIL import ImageGrab, Image
import numpy as np

# Windows API
import win32gui
import win32con
import win32api
import win32process

# 输入模拟
import keyboard
import mouse

# 配置文件路径
CONFIG_FILE = "config.ini"

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("client.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DNFAutoClient")

class DNFAutoClient:
    """DNF自动化客户端类"""
    
    def __init__(self):
        """初始化客户端"""
        self.config = self.load_config()
        self.server_url = self.config.get("Server", "url")
        self.client_id = None
        self.running = False
        self.ws = None
        self.capture_interval = float(self.config.get("Capture", "interval"))
        self.game_state = {"in_battle": False}
    
    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(CONFIG_FILE):
            self.create_default_config()
        
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE, encoding="utf-8")
        return config
    
    def create_default_config(self):
        """创建默认配置文件"""
        config = configparser.ConfigParser()
        
        config["Server"] = {
            "url": "wss://your-server-url:8080/ws",
            "verify_ssl": "false"
        }
        
        config["Capture"] = {
            "interval": "0.5",
            "quality": "70"
        }
        
        config["Game"] = {
            "window_title": "地下城与勇士",
            "key_mapping": "default"
        }
        
        # 保存配置
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            config.write(f)
        
        logger.info(f"已创建默认配置文件: {CONFIG_FILE}")
    
    async def connect(self):
        """连接到服务器"""
        logger.info(f"正在连接到服务器: {self.server_url}")
        
        ssl_context = None
        if self.server_url.startswith("wss://"):
            ssl_context = ssl.create_default_context()
            if self.config.get("Server", "verify_ssl").lower() == "false":
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            self.ws = await websockets.connect(
                self.server_url,
                ssl=ssl_context,
                max_size=10 * 1024 * 1024  # 10MB
            )
            
            # 等待认证
            await self.authenticate()
            
            logger.info("已连接到服务器")
            return True
            
        except Exception as e:
            logger.error(f"连接服务器失败: {e}")
            return False
    
    async def authenticate(self):
        """客户端认证"""
        try:
            # 等待认证挑战
            challenge_raw = await self.ws.recv()
            challenge = json.loads(challenge_raw)
            
            if challenge.get("type") != "auth_challenge":
                raise ValueError("无效的认证挑战")
            
            # 发送认证响应
            await self.ws.send(json.dumps({
                "type": "auth_response",
                "response": f"client_{random.getrandbits(32)}",
                "client_info": {
                    "version": "1.0.0",
                    "os": "Windows",
                    "screen_resolution": self.get_screen_resolution()
                }
            }))
            
            # 等待认证结果
            result_raw = await self.ws.recv()
            result = json.loads(result_raw)
            
            if result.get("type") != "auth_result" or result.get("status") != "success":
                raise ValueError("认证失败")
            
            # 保存客户端ID
            self.client_id = result.get("client_id")
            logger.info(f"认证成功，客户端ID: {self.client_id}")
            
        except Exception as e:
            logger.error(f"认证失败: {e}")
            raise
    
    def get_screen_resolution(self):
        """获取屏幕分辨率"""
        user32 = ctypes.windll.user32
        return [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    
    def get_game_window(self):
        """获取游戏窗口句柄"""
        window_title = self.config.get("Game", "window_title")
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd == 0:
            logger.warning(f"找不到游戏窗口: {window_title}")
            return None
        return hwnd
    
    def capture_game_screen(self, hwnd=None):
        """截取游戏画面"""
        try:
            if hwnd is None:
                hwnd = self.get_game_window()
                if hwnd is None:
                    return None
            
            # 获取窗口位置和大小
            rect = win32gui.GetWindowRect(hwnd)
            x, y, width, height = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
            
            # 截取画面
            screen = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            
            # 压缩图像
            quality = int(self.config.get("Capture", "quality"))
            buffer = BytesIO()
            screen.save(buffer, format="JPEG", quality=quality)
            
            # 转换为Base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return {
                "image": img_base64,
                "window_rect": [x, y, width, height],
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"截取游戏画面失败: {e}")
            return None
    
    async def send_heartbeat(self):
        """发送心跳包"""
        if not self.ws or not self.running:
            return
        
        try:
            await self.ws.send(json.dumps({
                "type": "heartbeat",
                "timestamp": time.time(),
                "client_id": self.client_id
            }))
            
            # 等待心跳响应
            response_raw = await asyncio.wait_for(
                self.ws.recv(),
                timeout=5.0
            )
            
            # 检查响应
            response = json.loads(response_raw)
            if response.get("type") != "heartbeat_response":
                logger.warning(f"收到非心跳响应: {response.get('type')}")
            
        except asyncio.TimeoutError:
            logger.warning("心跳超时")
        except Exception as e:
            logger.error(f"发送心跳失败: {e}")
    
    async def heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            await self.send_heartbeat()
            await asyncio.sleep(5.0)  # 每5秒发送一次心跳
    
    async def execute_action(self, action):
        """执行操作"""
        try:
            action_type = action.get("type")
            
            # 等待指定的延迟时间
            if "delay" in action:
                await asyncio.sleep(action["delay"])
            
            # 获取游戏窗口
            hwnd = self.get_game_window()
            if hwnd is None:
                return
            
            # 确保窗口处于前台
            if win32gui.GetForegroundWindow() != hwnd:
                win32gui.SetForegroundWindow(hwnd)
                await asyncio.sleep(0.1)
            
            # 获取窗口位置
            rect = win32gui.GetWindowRect(hwnd)
            window_x, window_y = rect[0], rect[1]
            
            # 执行不同类型的操作
            if action_type == "move_to":
                # 移动到指定位置
                position = action.get("position", [0, 0])
                x, y = position[0] + window_x, position[1] + window_y
                
                # 生成人类化的移动路径
                current_pos = win32gui.GetCursorPos()
                path = self.generate_movement_path(current_pos, [x, y])
                
                # 执行移动
                for point in path:
                    win32api.SetCursorPos((int(point[0]), int(point[1])))
                    await asyncio.sleep(0.01)  # 10ms延迟
            
            elif action_type == "use_skill":
                # 使用技能
                key = action.get("key", "1")
                keyboard.press(key)
                await asyncio.sleep(0.05)
                keyboard.release(key)
                
                # 如果有目标位置，移动鼠标并点击
                if "target_position" in action:
                    pos = action["target_position"]
                    x, y = pos[0] + window_x, pos[1] + window_y
                    win32api.SetCursorPos((int(x), int(y)))
                    await asyncio.sleep(0.05)
                    mouse.click()
            
            elif action_type == "interact":
                # 交互
                key = action.get("key", "f")
                keyboard.press(key)
                await asyncio.sleep(0.1)
                keyboard.release(key)
            
            elif action_type == "move_random":
                # 随机移动
                direction = action.get("direction", "right")
                duration = action.get("duration", 1.0)
                
                # 方向键映射
                dir_keys = {
                    "up": "w",
                    "down": "s",
                    "left": "a",
                    "right": "d"
                }
                
                key = dir_keys.get(direction, "d")
                keyboard.press(key)
                await asyncio.sleep(duration)
                keyboard.release(key)
            
            elif action_type == "use_item":
                # 使用物品
                key = action.get("key", "f1")
                keyboard.press(key)
                await asyncio.sleep(0.1)
                keyboard.release(key)
            
            elif action_type == "stop":
                # 停止所有按键
                for key in ["w", "a", "s", "d", "1", "2", "3", "4", "5", "6"]:
                    if keyboard.is_pressed(key):
                        keyboard.release(key)
            
            else:
                logger.warning(f"未知操作类型: {action_type}")
            
        except Exception as e:
            logger.error(f"执行操作失败: {e}")
    
    def generate_movement_path(self, start_pos, end_pos, steps=None):
        """生成模拟人类的鼠标移动路径"""
        # 计算距离
        distance = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
        
        # 如果未指定步数，根据距离计算
        if steps is None:
            steps = max(int(distance / 20), 5)  # 每20像素一个点，至少5个点
        
        # 生成基础路径
        t = np.linspace(0, 1, steps)
        path = []
        
        for i in range(steps):
            # 基础线性插值
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * t[i]
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * t[i]
            
            # 添加随机偏移（越靠近中间偏移越大）
            mid_factor = 4 * t[i] * (1 - t[i])  # 在中间最大
            max_offset = distance * 0.05 * mid_factor  # 最大偏移为距离的5%
            
            offset_x = random.normalvariate(0, max_offset / 3)
            offset_y = random.normalvariate(0, max_offset / 3)
            
            # 添加到路径
            path.append([x + offset_x, y + offset_y])
        
        # 确保起点和终点准确
        path[0] = start_pos
        path[-1] = end_pos
        
        return path
    
    async def capture_and_process_loop(self):
        """截图和处理循环"""
        while self.running:
            try:
                # 获取游戏窗口
                hwnd = self.get_game_window()
                if hwnd is None:
                    await asyncio.sleep(1.0)  # 找不到窗口时等待1秒
                    continue
                
                # 截取游戏画面
                screen_data = self.capture_game_screen(hwnd)
                if screen_data is None:
                    await asyncio.sleep(0.5)  # 截图失败时等待0.5秒
                    continue
                
                # 准备请求数据
                request = {
                    "type": "image",
                    "request_id": f"req_{random.getrandbits(32)}",
                    "timestamp": time.time(),
                    "data": screen_data["image"],
                    "game_state": self.game_state,
                    "window_rect": screen_data["window_rect"]
                }
                
                # 发送请求
                await self.ws.send(json.dumps(request))
                
                # 等待响应
                response_raw = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=5.0
                )
                
                # 处理响应
                response = json.loads(response_raw)
                
                if response.get("type") == "action_response":
                    # 执行动作
                    actions = response.get("actions", [])
                    
                    # 按优先级排序
                    actions.sort(key=lambda x: x.get("execution_priority", 1.0))
                    
                    for action in actions:
                        await self.execute_action(action)
                
                elif response.get("type") == "error":
                    logger.error(f"服务器返回错误: {response.get('message')}")
                
                # 等待指定的间隔时间
                await asyncio.sleep(self.capture_interval)
                
            except asyncio.TimeoutError:
                logger.warning("等待服务器响应超时")
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket连接已关闭")
                self.running = False
                break
            except Exception as e:
                logger.error(f"处理循环出错: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1.0)  # 出错时等待1秒
    
    async def run(self):
        """运行客户端"""
        self.running = True
        
        # 连接到服务器
        if not await self.connect():
            self.running = False
            return
        
        try:
            # 创建任务
            capture_task = asyncio.create_task(self.capture_and_process_loop())
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            
            # 等待任务完成
            await asyncio.gather(capture_task, heartbeat_task)
            
        except asyncio.CancelledError:
            logger.info("客户端任务已取消")
        except Exception as e:
            logger.error(f"客户端运行出错: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.running = False
            if self.ws:
                await self.ws.close()
            logger.info("客户端已停止")
    
    def start(self):
        """启动客户端"""
        asyncio.run(self.run())
    
    def stop(self):
        """停止客户端"""
        self.running = False
        logger.info("正在停止客户端...")

# 启动客户端
if __name__ == "__main__":
    try:
        client = DNFAutoClient()
        client.start()
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
    except Exception as e:
        logger.error(f"客户端出错: {e}")
        logger.error(traceback.format_exc())