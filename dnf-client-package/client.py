#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DNF自动化客户端，负责截取游戏画面并执行服务器返回的操作
优化版 - 增加断线重连、性能优化和增强的游戏状态管理
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
from PIL import Image
import numpy as np

# Windows API
import win32gui
import win32con
import win32api
import win32process

# 输入模拟
import keyboard
import mouse

# 尝试导入mss库，如果不存在则继续使用PIL
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("警告: mss库未安装，将使用PIL进行截图（性能较低）")
    print("请使用 pip install mss 安装以获得更好的性能")

# 配置文件路径 - 使用绝对路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "client.log"), encoding="utf-8"),
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
        self.max_retries = int(self.config.get("Connection", "max_retries", fallback="5"))
        self.retry_delay = int(self.config.get("Connection", "retry_delay", fallback="5"))
        
        # 增强的游戏状态
        self.game_state = {
            "in_battle": False,
            "current_map": "",
            "hp_percent": 100,
            "mp_percent": 100,
            "active_buffs": [],
            "cooldowns": {},
            "inventory_full": False,
            "current_quest": None,
            "last_operation_time": time.time(),
            "session_start_time": time.time()
        }
        
        # 连接状态
        self.last_heartbeat_time = 0
        self.connection_attempts = 0
        self.reconnecting = False
    
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
        
        config["Connection"] = {
            "max_retries": "5",
            "retry_delay": "5",
            "heartbeat_interval": "5"
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
                max_size=10 * 1024 * 1024,  # 10MB
                ping_interval=None,  # 禁用自动ping，我们将使用自己的心跳
                close_timeout=5
            )
            
            # 等待认证
            await self.authenticate()
            
            logger.info("已连接到服务器")
            self.connection_attempts = 0  # 重置连接尝试次数
            self.last_heartbeat_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"连接服务器失败: {e}")
            return False
    
    async def connect_with_retry(self):
        """带重试机制的连接函数"""
        if self.reconnecting:
            logger.info("已有重连过程在进行中，跳过")
            return False
            
        self.reconnecting = True
        retry_count = 0
        
        try:
            while retry_count < self.max_retries and not self.ws:
                try:
                    logger.info(f"尝试连接服务器 (尝试 {retry_count + 1}/{self.max_retries})...")
                    success = await self.connect()
                    if success:
                        self.reconnecting = False
                        return True
                except Exception as e:
                    logger.error(f"连接服务器失败: {e}")
                
                retry_count += 1
                retry_delay = min(60, self.retry_delay * (2 ** retry_count))  # 指数退避策略
                logger.info(f"等待 {retry_delay} 秒后重试...")
                await asyncio.sleep(retry_delay)
            
            if not self.ws:
                logger.error(f"达到最大重试次数 ({self.max_retries})，连接失败")
                self.reconnecting = False
                return False
        except Exception as e:
            logger.error(f"重连过程中发生错误: {e}")
            self.reconnecting = False
            return False
            
        self.reconnecting = False
        return True
    
    async def authenticate(self):
        """客户端认证"""
        try:
            # 等待认证挑战
            challenge_raw = await self.ws.recv()
            challenge = json.loads(challenge_raw)
            
            if challenge.get("type") != "auth_challenge":
                raise ValueError("无效的认证挑战")
            
            # 收集系统信息
            system_info = self.get_system_info()
            
            # 发送认证响应
            await self.ws.send(json.dumps({
                "type": "auth_response",
                "response": f"client_{random.getrandbits(32)}",
                "client_info": {
                    "version": "1.0.1",  # 更新版本号
                    "os": "Windows",
                    "screen_resolution": self.get_screen_resolution(),
                    "system_info": system_info
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
    
    def get_system_info(self):
        """获取系统信息"""
        system_info = {}
        try:
            system_info["hostname"] = os.environ.get("COMPUTERNAME", "Unknown")
            system_info["username"] = os.environ.get("USERNAME", "Unknown")
            system_info["processor"] = os.environ.get("PROCESSOR_IDENTIFIER", "Unknown")
            
            # 获取系统内存信息
            mem = ctypes.c_ulonglong()
            ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem))
            system_info["memory_gb"] = round(mem.value / (1024 * 1024), 2)
            
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
        
        return system_info
    
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
            
            # 检查窗口是否最小化
            if width <= 0 or height <= 0:
                logger.warning("游戏窗口被最小化或大小无效")
                return None
            
            # 使用mss进行截图(如果可用)，速度比PIL快5-10倍
            if MSS_AVAILABLE:
                with mss.mss() as sct:
                    monitor = {"top": y, "left": x, "width": width, "height": height}
                    sct_img = sct.grab(monitor)
                    # 将mss图像转换为PIL图像
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            else:
                # 使用PIL进行截图(备选方案)
                img = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            
            # 压缩图像
            quality = int(self.config.get("Capture", "quality"))
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            
            # 转换为Base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return {
                "image": img_base64,
                "window_rect": [x, y, width, height],
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"截取游戏画面失败: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def send_heartbeat(self):
        """发送心跳包"""
        if not self.ws or not self.running:
            return False
        
        try:
            await self.ws.send(json.dumps({
                "type": "heartbeat",
                "timestamp": time.time(),
                "client_id": self.client_id,
                "game_state": {
                    "in_battle": self.game_state["in_battle"],
                    "current_map": self.game_state["current_map"]
                }
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
                return False
            
            self.last_heartbeat_time = time.time()
            return True
            
        except asyncio.TimeoutError:
            logger.warning("心跳超时")
            return False
        except websockets.exceptions.ConnectionClosed:
            logger.warning("发送心跳时连接已关闭")
            return False
        except Exception as e:
            logger.error(f"发送心跳失败: {e}")
            return False
    
    async def heartbeat_loop(self):
        """心跳循环"""
        heartbeat_interval = float(self.config.get("Connection", "heartbeat_interval", fallback="5"))
        
        while self.running:
            if self.ws and not self.ws.closed:
                success = await self.send_heartbeat()
                if not success:
                    # 心跳失败，检查连接状态
                    if time.time() - self.last_heartbeat_time > heartbeat_interval * 3:
                        logger.warning(f"心跳超时超过 {heartbeat_interval * 3} 秒，尝试重新连接")
                        if self.ws:
                            await self.ws.close()
                            self.ws = None
            
            await asyncio.sleep(heartbeat_interval)
    
    async def reconnect_loop(self):
        """重连检查循环"""
        while self.running:
            if not self.ws or self.ws.closed:
                logger.warning("WebSocket连接已断开，尝试重连...")
                if await self.connect_with_retry():
                    logger.info("重连成功")
                else:
                    logger.error("重连失败")
                    # 不要立即停止，继续尝试
            
            await asyncio.sleep(5)  # 每5秒检查一次连接状态
    
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
                try:
                    # 尝试多种方法激活窗口
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # 恢复窗口（如果最小化）
                    win32gui.SetForegroundWindow(hwnd)  # 尝试置为前台
                    await asyncio.sleep(0.1)
                    
                    # 如果窗口仍然不在前台，使用更强的方法
                    if win32gui.GetForegroundWindow() != hwnd:
                        # 获取当前激活窗口的线程和进程ID
                        curr_hwnd = win32gui.GetForegroundWindow()
                        curr_thread_id = win32process.GetWindowThreadProcessId(curr_hwnd)[0]
                        # 获取目标窗口的线程和进程ID
                        target_thread_id = win32process.GetWindowThreadProcessId(hwnd)[0]
                        # 附加线程输入
                        win32process.AttachThreadInput(target_thread_id, curr_thread_id, True)
                        win32gui.SetForegroundWindow(hwnd)
                        win32gui.BringWindowToTop(hwnd)
                        win32process.AttachThreadInput(target_thread_id, curr_thread_id, False)
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"激活窗口失败: {e}")
            
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
            
            elif action_type == "click":
                # 点击指定位置
                position = action.get("position", [0, 0])
                x, y = position[0] + window_x, position[1] + window_y
                
                # 移动到位置
                current_pos = win32gui.GetCursorPos()
                path = self.generate_movement_path(current_pos, [x, y])
                
                for point in path:
                    win32api.SetCursorPos((int(point[0]), int(point[1])))
                    await asyncio.sleep(0.01)
                
                # 执行点击
                mouse.click()
            
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
                        
            elif action_type == "type_text":
                # 输入文本
                text = action.get("text", "")
                if text:
                    keyboard.write(text, delay=0.05)
                    
            elif action_type == "press_key_combo":
                # 按下组合键
                keys = action.get("keys", [])
                if keys:
                    for key in keys:
                        keyboard.press(key)
                    await asyncio.sleep(0.1)
                    for key in reversed(keys):
                        keyboard.release(key)
            
            else:
                logger.warning(f"未知操作类型: {action_type}")
            
            # 更新游戏状态
            self.game_state["last_operation_time"] = time.time()
            
        except Exception as e:
            logger.error(f"执行操作失败: {e}")
            logger.error(traceback.format_exc())
    
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
        
        # 动态调整平滑度
        smoothness = float(self.config.get("Game", "movement_smoothness", fallback="0.8"))
        
        for i in range(steps):
            # 基础线性插值
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * t[i]
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * t[i]
            
            # 添加随机偏移（越靠近中间偏移越大）
            mid_factor = 4 * t[i] * (1 - t[i])  # 在中间最大
            max_offset = distance * 0.05 * mid_factor * (1 - smoothness)  # 最大偏移为距离的5%，受平滑度影响
            
            offset_x = random.normalvariate(0, max_offset / 3)
            offset_y = random.normalvariate(0, max_offset / 3)
            
            # 添加到路径
            path.append([x + offset_x, y + offset_y])
        
        # 确保起点和终点准确
        path[0] = start_pos
        path[-1] = end_pos
        
        return path
    
    def analyze_image(self, image_data, detection_results):
        """
        分析图像数据，更新游戏状态
        
        参数:
            image_data (dict): 图像数据
            detection_results (list): 检测结果
        """
        # 更新战斗状态
        monsters_detected = False
        for det in detection_results:
            if det["class_name"] in ["monster", "boss"]:
                monsters_detected = True
                break
        
        self.game_state["in_battle"] = monsters_detected
        
        # 分析血条和蓝条
        hp_bars = [d for d in detection_results if d["class_name"] == "hp_bar"]
        mp_bars = [d for d in detection_results if d["class_name"] == "mp_bar"]
        
        if hp_bars:
            # 估算血量百分比
            self.game_state["hp_percent"] = self.estimate_bar_percent(hp_bars[0])
        
        if mp_bars:
            # 估算蓝量百分比
            self.game_state["mp_percent"] = self.estimate_bar_percent(mp_bars[0])
        
        # 检测技能冷却
        cooldowns = [d for d in detection_results if d["class_name"] == "cooldown"]
        self.game_state["cooldowns"] = {}
        
        for cd in cooldowns:
            if "skill_id" in cd:
                self.game_state["cooldowns"][cd["skill_id"]] = cd.get("remaining_time", 1.0)
    
    def estimate_bar_percent(self, bar_detection):
        """
        估计血条/蓝条的百分比
        
        参数:
            bar_detection (dict): 条形检测结果
            
        返回:
            float: 百分比值(0-100)
        """
        # 这里需要根据实际情况实现
        # 临时方案：返回检测结果中的值，如果没有则返回默认值
        return bar_detection.get("percent", 100)
    
    async def capture_and_process_loop(self):
        """截图和处理循环"""
        consecutive_errors = 0
        
        while self.running:
            try:
                # 检查WebSocket连接
                if not self.ws or self.ws.closed:
                    await asyncio.sleep(0.5)  # 连接断开时等待
                    continue
                
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
                    # 获取检测结果并更新游戏状态
                    if "detections" in response:
                        self.analyze_image(screen_data, response["detections"])
                    
                    # 执行动作
                    actions = response.get("actions", [])
                    
                    # 按优先级排序
                    actions.sort(key=lambda x: x.get("execution_priority", 1.0))
                    
                    for action in actions:
                        await self.execute_action(action)
                    
                    # 重置错误计数
                    consecutive_errors = 0
                
                elif response.get("type") == "error":
                    logger.error(f"服务器返回错误: {response.get('message')}")
                    consecutive_errors += 1
                
                # 等待指定的间隔时间
                await asyncio.sleep(self.capture_interval)
                
            except asyncio.TimeoutError:
                logger.warning("等待服务器响应超时")
                consecutive_errors += 1
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket连接已关闭")
                break
            except Exception as e:
                logger.error(f"处理循环出错: {e}")
                logger.error(traceback.format_exc())
                consecutive_errors += 1
                await asyncio.sleep(1.0)  # 出错时等待1秒
            
            # 如果连续错误过多，尝试重新连接
            if consecutive_errors >= 5:
                logger.warning(f"连续出错 {consecutive_errors} 次，尝试重新连接")
                if self.ws:
                    await self.ws.close()
                    self.ws = None
                consecutive_errors = 0
    
    async def run(self):
        """运行客户端"""
        self.running = True
        
        # 连接到服务器
        if not await self.connect_with_retry():
            self.running = False
            return
        
        try:
            # 创建任务
            capture_task = asyncio.create_task(self.capture_and_process_loop())
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            reconnect_task = asyncio.create_task(self.reconnect_loop())
            
            # 等待任务完成
            await asyncio.gather(capture_task, heartbeat_task, reconnect_task)
            
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
        try:
            # 启动心跳监控线程（备用方案，以防异步心跳失效）
            self._monitor_thread = threading.Thread(target=self._monitor_connection)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            
            # 运行主循环
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("用户中断，正在退出...")
        except Exception as e:
            logger.error(f"客户端出错: {e}")
            logger.error(traceback.format_exc())
    
    def _monitor_connection(self):
        """监控连接的后台线程"""
        while True:
            try:
                time.sleep(30)  # 每30秒检查一次
                
                if not self.running:
                    break
                    
                # 检查心跳时间
                if self.last_heartbeat_time > 0 and time.time() - self.last_heartbeat_time > 60:
                    logger.warning("心跳超时，可能需要重连")
                    # 不直接重连，留给重连循环处理
            except Exception as e:
                logger.error(f"连接监控线程出错: {e}")
    
    def stop(self):
        """停止客户端"""
        self.running = False
        logger.info("正在停止客户端...")

# 创建默认配置文件（如果不存在）
def ensure_config():
    if not os.path.exists(CONFIG_FILE):
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
            "key_mapping": "default",
            "movement_smoothness": "0.8"
        }
        
        config["Connection"] = {
            "max_retries": "5",
            "retry_delay": "5",
            "heartbeat_interval": "5"
        }
        
        # 保存配置
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            config.write(f)
        
        print(f"已创建默认配置文件: {CONFIG_FILE}")
        print("请编辑配置文件设置正确的服务器地址等信息")

# 启动客户端
if __name__ == "__main__":
    # 确保配置文件存在
    ensure_config()
    
    try:
        client = DNFAutoClient()
        client.start()
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
    except Exception as e:
        logger.error(f"客户端出错: {e}")
        logger.error(traceback.format_exc())