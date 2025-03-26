#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人类行为模拟工具，提供模拟人类操作的功能
"""

import random
import time
import math
import numpy as np
from config.settings import BEHAVIOR

def generate_human_delay():
    """
    生成人类化的延迟时间
    
    返回:
        float: 延迟时间（秒）
    """
    # 使用正态分布生成随机延迟
    mean_delay = (BEHAVIOR["min_delay"] + BEHAVIOR["max_delay"]) / 2
    std_dev = (BEHAVIOR["max_delay"] - BEHAVIOR["min_delay"]) / 6  # 使99.7%的值在范围内
    
    delay = random.normalvariate(mean_delay, std_dev)
    
    # 确保在设定范围内
    return max(BEHAVIOR["min_delay"], min(BEHAVIOR["max_delay"], delay))

def generate_human_movement_path(start_pos, end_pos, steps=None):
    """
    生成模拟人类的鼠标移动路径
    
    参数:
        start_pos (list): 起始位置 [x, y]
        end_pos (list): 目标位置 [x, y]
        steps (int): 路径点数量，如果为None则自动计算
        
    返回:
        list: 路径点列表 [[x1, y1], [x2, y2], ...]
    """
    # 计算距离
    distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
    
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
    path[0] = start_pos.copy()
    path[-1] = end_pos.copy()
    
    return path

def generate_human_click_offset(target_pos, target_size=(50, 50)):
    """
    生成人类化的点击位置偏移
    
    参数:
        target_pos (list): 目标中心位置 [x, y]
        target_size (tuple): 目标大小 (width, height)
        
    返回:
        list: 带偏移的点击位置 [x, y]
    """
    # 计算最大偏移（不超过目标大小的一半）
    max_offset_x = min(target_size[0] / 2 * 0.8, 10)  # 最大偏移不超过10像素
    max_offset_y = min(target_size[1] / 2 * 0.8, 10)
    
    # 生成偏移（中心位置概率更高）
    offset_x = random.normalvariate(0, max_offset_x / 3)
    offset_y = random.normalvariate(0, max_offset_y / 3)
    
    # 应用偏移
    click_pos = [
        target_pos[0] + offset_x,
        target_pos[1] + offset_y
    ]
    
    return click_pos

def generate_typing_speed(base_cps=5.0, variance=0.2):
    """
    生成人类化的打字速度
    
    参数:
        base_cps (float): 基础字符每秒速度
        variance (float): 速度方差
        
    返回:
        float: 字符间延迟时间（秒）
    """
    # 添加随机波动
    speed = random.normalvariate(base_cps, base_cps * variance)
    speed = max(speed, base_cps * 0.5)  # 确保速度不会太慢
    
    # 将速度转换为延迟
    delay = 1.0 / speed
    
    return delay