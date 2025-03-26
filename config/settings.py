#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全局配置参数
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 模型配置
MODEL = {
    "name": "yolov8m",
    "weights": os.path.join(BASE_DIR, "models", "weights", "dnf_yolo8m.pt"),
    "conf_threshold": 0.5,
    "iou_threshold": 0.45,
    "device": "cuda"  # 使用GPU
}

# 服务器配置
SERVER = {
    "max_connections": 10,
    "timeout": 60,
    "heartbeat_interval": 5
}

# 安全配置
SECURITY = {
    "token_expiry": 3600,  # 1小时
    "encryption_enabled": True,
    "ssl_enabled": True,
    "ssl_cert": os.path.join(BASE_DIR, "config", "cert.pem"),
    "ssl_key": os.path.join(BASE_DIR, "config", "key.pem")
}

# 行为模拟配置
BEHAVIOR = {
    "min_delay": 0.1,
    "max_delay": 0.8,
    "click_variance": 0.15,  # 点击位置随机偏移比例
    "movement_smoothness": 0.8  # 移动平滑度 (0-1)
}

# 日志配置
LOGGING = {
    "level": "INFO",
    "file": os.path.join(BASE_DIR, "data", "logs", "server.log"),
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# DNF游戏特定配置
DNF = {
    "classes": [
        "monster", "boss", "door", "item", "npc", "player", 
        "hp_bar", "mp_bar", "skill_ready", "cooldown"
    ],
    "maps": {
        "赫顿玛尔": {"entrance": (123, 456), "exit": (789, 101)},
        "天空之城": {"entrance": (234, 567), "exit": (890, 121)},
        # 更多地图配置...
    },
    "skills": {
        "1": "普通攻击",
        "2": "技能1",
        "3": "技能2",
        # 更多技能配置...
    }
}