#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人类行为模拟工具，提供模拟人类操作的功能
优化版 - 更自然的行为模式和个性化特征
"""

import random
import time
import math
import numpy as np
from datetime import datetime
from config.settings import BEHAVIOR

# 不同用户的行为模式
USER_PROFILES = {
    "fast": {
        "delay_mean": 0.15,
        "delay_std": 0.05,
        "movement_smoothness": 0.7,
        "click_accuracy": 0.9,
        "double_click_chance": 0.05,
        "description": "快速玩家 - 反应迅速，操作精准"
    },
    "normal": {
        "delay_mean": 0.25,
        "delay_std": 0.1,
        "movement_smoothness": 0.8,
        "click_accuracy": 0.85,
        "double_click_chance": 0.1,
        "description": "普通玩家 - 中等速度，操作稳定"
    },
    "casual": {
        "delay_mean": 0.4,
        "delay_std": 0.15,
        "movement_smoothness": 0.6,
        "click_accuracy": 0.7,
        "double_click_chance": 0.15,
        "description": "休闲玩家 - 操作较慢，精确度一般"
    }
}

# 当前用户配置文件
current_profile = "normal"

# 用户注意力随时间变化的模拟
attention_level = 1.0
last_attention_update = time.time()

# 活动模式（早上、下午、晚上）
activity_mode = "normal"

def set_user_profile(profile_name):
    """
    设置用户行为配置文件
    
    参数:
        profile_name (str): 配置文件名称 ("fast", "normal", "casual")
    """
    global current_profile
    if profile_name in USER_PROFILES:
        current_profile = profile_name
        return True
    return False

def update_attention_level():
    """更新用户注意力水平，模拟人类随时间的疲劳"""
    global attention_level, last_attention_update
    
    current_time = time.time()
    elapsed = current_time - last_attention_update
    
    # 随着时间推移，注意力逐渐下降
    if elapsed > 60:  # 每分钟更新一次
        # 注意力随时间缓慢下降（0.5%/分钟）
        attention_decay = 0.005 * (elapsed / 60)
        
        # 随机波动（+/- 2%）
        attention_fluctuation = random.uniform(-0.02, 0.02)
        
        # 更新注意力水平
        attention_level = max(0.7, min(1.0, attention_level - attention_decay + attention_fluctuation))
        
        # 重置时间戳
        last_attention_update = current_time
    
    return attention_level

def set_activity_mode():
    """根据一天中的时间设置活动模式"""
    global activity_mode
    
    # 获取当前小时
    current_hour = datetime.now().hour
    
    # 根据时间段设置模式
    if 5 <= current_hour < 9:  # 早晨
        activity_mode = "morning"
    elif 9 <= current_hour < 17:  # 工作时间
        activity_mode = "normal"
    elif 17 <= current_hour < 22:  # 晚上
        activity_mode = "evening"
    else:  # 深夜
        activity_mode = "night"
    
    return activity_mode

def generate_human_delay():
    """
    生成人类化的延迟时间 - 优化版
    
    返回:
        float: 延迟时间（秒）
    """
    # 获取当前用户配置
    profile = USER_PROFILES[current_profile]
    
    # 更新注意力水平
    attention = update_attention_level()
    
    # 检查活动模式
    mode = set_activity_mode()
    
    # 基础延迟参数
    delay_mean = profile["delay_mean"]
    delay_std = profile["delay_std"]
    
    # 根据注意力调整
    # 注意力下降，延迟增加
    delay_mean = delay_mean * (2 - attention)
    
    # 根据活动模式调整
    if mode == "morning":
        # 早晨略慢
        delay_mean *= 1.1
    elif mode == "evening":
        # 晚上正常
        pass
    elif mode == "night":
        # 深夜反应较慢
        delay_mean *= 1.2
        delay_std *= 1.2
    
    # 偶尔的停顿（思考时间）
    if random.random() < 0.05:  # 5%几率
        thinking_time = random.uniform(0.5, 1.5)
        return delay_mean + thinking_time
    
    # 使用正态分布生成随机延迟
    delay = random.normalvariate(delay_mean, delay_std)
    
    # 确保在合理范围内
    min_delay = BEHAVIOR.get("min_delay", 0.1)
    max_delay = BEHAVIOR.get("max_delay", 0.8)
    return max(min_delay, min(max_delay, delay))

def generate_human_movement_path(start_pos, end_pos, steps=None):
    """
    生成模拟人类的鼠标移动路径 - 优化版
    
    参数:
        start_pos (list): 起始位置 [x, y]
        end_pos (list): 目标位置 [x, y]
        steps (int): 路径点数量，如果为None则自动计算
        
    返回:
        list: 路径点列表 [[x1, y1], [x2, y2], ...]
    """
    # 获取当前用户配置
    profile = USER_PROFILES[current_profile]
    smoothness = profile["movement_smoothness"]
    
    # 计算距离
    distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
    
    # 如果距离太短，使用直线路径
    if distance < 10:
        return [start_pos, end_pos]
    
    # 如果未指定步数，根据距离和速度计算
    if steps is None:
        # 根据距离、平滑度和注意力计算步数
        attention = update_attention_level()
        base_steps = max(int(distance / 20), 5)  # 每20像素一个点，至少5个点
        
        # 平滑度越高，步数越多；注意力越高，步数越多
        steps = int(base_steps * smoothness * attention)
        steps = max(5, min(30, steps))  # 限制在合理范围内
    
    # 生成基础路径
    t = np.linspace(0, 1, steps)
    path = []
    
    # 添加曲率 - 模拟手腕运动
    # 贝塞尔曲线的控制点
    control_x = start_pos[0] + (end_pos[0] - start_pos[0]) / 2
    control_y = start_pos[1] + (end_pos[1] - start_pos[1]) / 2
    
    # 添加随机偏移到控制点
    offset_factor = (1.0 - smoothness) * 0.5  # 平滑度越低，偏移越大
    max_offset = distance * offset_factor
    control_x += random.normalvariate(0, max_offset / 2)
    control_y += random.normalvariate(0, max_offset / 2)
    
    for i in range(steps):
        # 二次贝塞尔曲线
        t_i = t[i]
        x = (1 - t_i)**2 * start_pos[0] + 2 * (1 - t_i) * t_i * control_x + t_i**2 * end_pos[0]
        y = (1 - t_i)**2 * start_pos[1] + 2 * (1 - t_i) * t_i * control_y + t_i**2 * end_pos[1]
        
        # 添加细微的随机抖动（手部微小颤抖）
        jitter_factor = (1.0 - smoothness) * 0.02 * distance
        jitter_x = random.normalvariate(0, jitter_factor)
        jitter_y = random.normalvariate(0, jitter_factor)
        
        # 注意力越低，抖动越大
        attention = update_attention_level()
        jitter_x *= (2 - attention)
        jitter_y *= (2 - attention)
        
        # 添加到路径
        path.append([x + jitter_x, y + jitter_y])
    
    # 确保起点和终点准确
    path[0] = start_pos.copy()
    path[-1] = end_pos.copy()
    
    # 模拟加速和减速
    # 路径采样 - 开始慢，中间快，结束慢
    if steps > 10:
        resampled_path = []
        resampled_path.append(path[0])  # 起点
        
        # 前20%缓慢加速
        accel_end = int(steps * 0.2)
        for i in range(1, accel_end):
            resampled_path.append(path[i])
        
        # 中间60%快速移动（跳过一些点）
        mid_start = accel_end
        mid_end = int(steps * 0.8)
        
        # 根据平滑度决定中间部分的采样率
        skip_factor = int(2 + (1 - smoothness) * 3)
        for i in range(mid_start, mid_end, skip_factor):
            resampled_path.append(path[min(i, steps - 1)])
        
        # 最后20%缓慢减速
        for i in range(mid_end, steps):
            resampled_path.append(path[i])
        
        if resampled_path[-1] != end_pos:
            resampled_path.append(end_pos)  # 确保终点
        
        return resampled_path
    
    return path

def generate_human_click_offset(target_pos, target_size=(50, 50)):
    """
    生成人类化的点击位置偏移 - 优化版
    
    参数:
        target_pos (list): 目标中心位置 [x, y]
        target_size (tuple): 目标大小 (width, height)
        
    返回:
        list: 带偏移的点击位置 [x, y]
    """
    # 获取当前用户配置
    profile = USER_PROFILES[current_profile]
    accuracy = profile["click_accuracy"]
    
    # 更新注意力水平
    attention = update_attention_level()
    
    # 综合准确度和注意力
    effective_accuracy = accuracy * attention
    
    # 计算最大偏移（不超过目标大小的一半）
    max_offset_x = min(target_size[0] / 2 * (1 - effective_accuracy) * 1.5, 15)
    max_offset_y = min(target_size[1] / 2 * (1 - effective_accuracy) * 1.5, 15)
    
    # 生成偏移（中心位置概率更高）
    offset_x = random.normalvariate(0, max_offset_x / 2)
    offset_y = random.normalvariate(0, max_offset_y / 2)
    
    # 应用偏移
    click_pos = [
        target_pos[0] + offset_x,
        target_pos[1] + offset_y
    ]
    
    return click_pos

def should_double_click():
    """
    确定是否应该双击
    
    返回:
        bool: 是否应该双击
    """
    profile = USER_PROFILES[current_profile]
    return random.random() < profile["double_click_chance"]

def generate_typing_speed(text_length, base_cps=5.0):
    """
    生成人类化的打字速度序列
    
    参数:
        text_length (int): 文本长度
        base_cps (float): 基础字符每秒速度
        
    返回:
        list: 每个字符的延迟时间列表
    """
    # 获取当前用户配置和注意力
    profile = USER_PROFILES[current_profile]
    attention = update_attention_level()
    
    # 基础打字速度 - 根据配置文件调整
    if current_profile == "fast":
        base_cps = 8.0
    elif current_profile == "casual":
        base_cps = 3.5
    
    # 根据注意力调整速度
    base_cps = base_cps * attention
    
    # 生成每个字符的延迟
    delays = []
    current_speed = base_cps
    
    for i in range(text_length):
        # 随机波动
        speed_variance = 0.2 * base_cps
        current_speed = random.normalvariate(base_cps, speed_variance)
        current_speed = max(base_cps * 0.5, current_speed)  # 确保速度不会太慢
        
        # 将速度转换为延迟
        delay = 1.0 / current_speed
        
        # 某些特殊情况
        if random.random() < 0.05:  # 偶尔停顿
            delay += random.uniform(0.2, 0.7)
        
        if i > 0 and i % 10 == 0 and random.random() < 0.2:  # 句子结束停顿
            delay += random.uniform(0.3, 0.8)
        
        delays.append(delay)
    
    return delays

def generate_action_sequence(action_count):
    """
    生成一系列人类化的动作延迟
    
    参数:
        action_count (int): 动作数量
        
    返回:
        list: 延迟时间列表
    """
    delays = []
    
    for i in range(action_count):
        # 基础延迟
        delay = generate_human_delay()
        
        # 连续动作的相关性
        if i > 0:
            # 连续动作通常更快
            delay *= 0.8
        
        # 序列中的变化
        if i == 0:
            # 第一个动作前可能有更长的准备时间
            delay *= 1.5
        elif i == action_count - 1:
            # 最后一个动作后可能有更长的思考时间
            delay *= 1.2
        
        delays.append(delay)
    
    return delays

def simulate_fatigue(session_duration):
    """
    模拟随时间产生的疲劳效应
    
    参数:
        session_duration (float): 会话持续时间（秒）
        
    返回:
        float: 疲劳系数 (1.0=正常, >1.0=疲劳)
    """
    # 基础疲劳曲线
    hours = session_duration / 3600.0
    
    # 前1小时基本无疲劳
    if hours < 1:
        base_fatigue = 1.0
    # 1-2小时逐渐增加疲劳
    elif hours < 2:
        base_fatigue = 1.0 + (hours - 1) * 0.1
    # 2-3小时疲劳加重
    elif hours < 3:
        base_fatigue = 1.1 + (hours - 2) * 0.2
    # 3小时以上疲劳显著
    else:
        base_fatigue = 1.3 + min(0.4, (hours - 3) * 0.1)
    
    # 添加随机波动
    fatigue = base_fatigue * random.uniform(0.95, 1.05)
    
    return fatigue

def get_behavior_profile():
    """
    获取当前行为配置文件信息
    
    返回:
        dict: 行为配置信息
    """
    profile = USER_PROFILES[current_profile].copy()
    profile["attention"] = update_attention_level()
    profile["activity_mode"] = set_activity_mode()
    
    return profile