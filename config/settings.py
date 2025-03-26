#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全局配置参数
优化版 - 增加更多配置选项和兼容性检查
"""

import os
import sys
import json
import torch
from pathlib import Path

# 检测当前环境
IS_LINUX = sys.platform.startswith('linux')
IS_WINDOWS = sys.platform.startswith('win')
HAS_CUDA = torch.cuda.is_available()

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 版本信息
VERSION = "1.0.1"
BUILD_DATE = "2025-03-27"

# 加载环境变量
def load_env_var(name, default):
    """从环境变量加载设置，如果不存在则使用默认值"""
    return os.environ.get(f"DNF_{name}", default)

# 模型配置
MODEL = {
    "name": load_env_var("MODEL_NAME", "yolov8m"),
    "weights": os.path.join(BASE_DIR, "models", "weights", load_env_var("MODEL_FILE", "dnf_yolo8m.pt")),
    "conf_threshold": float(load_env_var("CONF_THRESHOLD", "0.5")),
    "iou_threshold": float(load_env_var("IOU_THRESHOLD", "0.45")),
    "device": load_env_var("DEVICE", "cuda" if HAS_CUDA else "cpu"),
    "half_precision": load_env_var("HALF_PRECISION", "true").lower() == "true" and HAS_CUDA,
    "batch_size": int(load_env_var("BATCH_SIZE", "1")),
    "img_size": int(load_env_var("IMG_SIZE", "640")),
    "engine": load_env_var("ENGINE", "pytorch"),  # pytorch 或 onnx
    "class_names": os.path.join(BASE_DIR, "data", "training", "data.yaml")
}

# 服务器配置
SERVER = {
    "max_connections": int(load_env_var("MAX_CONNECTIONS", "10")),
    "timeout": int(load_env_var("TIMEOUT", "60")),
    "heartbeat_interval": int(load_env_var("HEARTBEAT_INTERVAL", "5")),
    "inactive_timeout": int(load_env_var("INACTIVE_TIMEOUT", "300")),  # 用户不活跃超时（秒）
    "max_retries": int(load_env_var("MAX_RETRIES", "3")),  # 操作最大重试次数
    "status_update_interval": int(load_env_var("STATUS_UPDATE_INTERVAL", "30")),  # 状态更新间隔（秒）
    "api_enabled": load_env_var("API_ENABLED", "true").lower() == "true"  # 是否启用API
}

# 安全配置
SECURITY = {
    "token_expiry": int(load_env_var("TOKEN_EXPIRY", "3600")),  # 1小时
    "encryption_enabled": False,  # 禁用加密以简化调试
    "ssl_enabled": False,  # 禁用SSL
    "ssl_cert": os.path.join(BASE_DIR, "config", "cert.pem"),
    "ssl_key": os.path.join(BASE_DIR, "config", "key.pem"),
    "ip_whitelist": load_env_var("IP_WHITELIST", "").split(",") if load_env_var("IP_WHITELIST", "") else [],
    "rate_limit": int(load_env_var("RATE_LIMIT", "100")),  # 每分钟最大请求数
    "authentication_required": False  # 暂时禁用认证要求
}

# 行为模拟配置
BEHAVIOR = {
    "min_delay": float(load_env_var("MIN_DELAY", "0.1")),
    "max_delay": float(load_env_var("MAX_DELAY", "0.8")),
    "click_variance": float(load_env_var("CLICK_VARIANCE", "0.15")),  # 点击位置随机偏移比例
    "movement_smoothness": float(load_env_var("MOVEMENT_SMOOTHNESS", "0.8")),  # 移动平滑度 (0-1)
    "double_click_chance": float(load_env_var("DOUBLE_CLICK_CHANCE", "0.05")),  # 双击概率
    "user_profile": load_env_var("USER_PROFILE", "normal"),  # 用户行为配置 (fast, normal, casual)
    "randomize_behavior": load_env_var("RANDOMIZE_BEHAVIOR", "true").lower() == "true"  # 是否随机化行为
}

# 日志配置
LOGGING = {
    "level": load_env_var("LOG_LEVEL", "INFO"),
    "file": os.path.join(BASE_DIR, "data", "logs", "server.log"),
    "max_size": int(load_env_var("LOG_MAX_SIZE", "10")) * 1024 * 1024,  # 10MB
    "backup_count": int(load_env_var("LOG_BACKUP_COUNT", "5")),
    "console_output": load_env_var("LOG_CONSOLE", "true").lower() == "true",
    "log_requests": load_env_var("LOG_REQUESTS", "true").lower() == "true",
    "log_performance": load_env_var("LOG_PERFORMANCE", "true").lower() == "true"
}

# 缓存配置
CACHE = {
    "enabled": load_env_var("CACHE_ENABLED", "true").lower() == "true",
    "directory": os.path.join(BASE_DIR, "data", "cache"),
    "max_size": int(load_env_var("CACHE_MAX_SIZE", "1000")),  # 最大缓存项数
    "ttl": int(load_env_var("CACHE_TTL", "3600"))  # 缓存有效期（秒）
}

# 会话配置
SESSION = {
    "directory": os.path.join(BASE_DIR, "data", "sessions"),
    "save_interval": int(load_env_var("SESSION_SAVE_INTERVAL", "300")),  # 会话保存间隔（秒）
    "max_idle_time": int(load_env_var("SESSION_MAX_IDLE", "1800")),  # 最大空闲时间（秒）
    "persistence_enabled": load_env_var("SESSION_PERSISTENCE", "true").lower() == "true"
}

# DNF游戏特定配置
DNF = {
    "classes": [
        "monster", "boss", "door", "item", "npc", "player", 
        "hp_bar", "mp_bar", "skill_ready", "cooldown", "dialog",
        "dialog_option", "revive_button", "confirm_button",
        "dungeon_portal", "dungeon_select", "dungeon_option",
        "room_select", "room_option", "death_ui"
    ],
    "maps": {
        "赫顿玛尔": {"entrance": (123, 456), "exit": (789, 101)},
        "天空之城": {"entrance": (234, 567), "exit": (890, 121)},
        "格兰之森": {"entrance": (345, 678), "exit": (901, 234)},
        "诺斯玛尔": {"entrance": (456, 789), "exit": (345, 678)},
        # 更多地图配置...
    },
    "skills": {
        "1": {"name": "普通攻击", "type": "melee", "cooldown": 0.5},
        "2": {"name": "技能1", "type": "melee", "cooldown": 3.0},
        "3": {"name": "技能2", "type": "ranged", "cooldown": 5.0},
        "4": {"name": "技能3", "type": "aoe", "cooldown": 8.0},
        "5": {"name": "技能4", "type": "buff", "cooldown": 15.0},
        "6": {"name": "技能5", "type": "ultimate", "cooldown": 30.0},
        # 更多技能配置...
    },
    "difficulty_levels": {
        "normal": {"monster_hp": 100, "monster_damage": 100},
        "hard": {"monster_hp": 150, "monster_damage": 150},
        "expert": {"monster_hp": 200, "monster_damage": 200},
        "master": {"monster_hp": 250, "monster_damage": 250},
        "hell": {"monster_hp": 300, "monster_damage": 300}
    }
}

# 检查并创建必要的目录
def ensure_directories():
    """确保所有需要的目录都存在"""
    directories = [
        os.path.dirname(LOGGING["file"]),  # 日志目录
        CACHE["directory"],  # 缓存目录
        SESSION["directory"],  # 会话目录
        os.path.join(BASE_DIR, "data", "training"),  # 训练数据目录
        os.path.dirname(MODEL["weights"]),  # 模型权重目录
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 配置验证
def validate_config():
    """验证配置有效性"""
    # 检查模型配置
    if not os.path.exists(MODEL["weights"]) and not os.path.exists(MODEL["weights"] + ".onnx"):
        print(f"警告: 模型权重文件不存在: {MODEL['weights']}")
    
    # 检查CUDA设置
    if MODEL["device"] == "cuda" and not HAS_CUDA:
        print("警告: 已配置使用CUDA但系统中没有可用的GPU，将使用CPU模式")
        MODEL["device"] = "cpu"
        MODEL["half_precision"] = False
    
    # 检查日志级别
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if LOGGING["level"] not in valid_log_levels:
        print(f"警告: 无效的日志级别 '{LOGGING['level']}'，将使用 'INFO'")
        LOGGING["level"] = "INFO"
    
    # 保存验证后的配置
    save_validated_config()

def save_validated_config():
    """保存验证后的配置到文件，便于其他进程读取"""
    config = {
        "version": VERSION,
        "build_date": BUILD_DATE,
        "model": MODEL,
        "server": SERVER,
        "security": {k: v for k, v in SECURITY.items() if k not in ["ssl_key"]}  # 不保存敏感信息
    }
    
    try:
        with open(os.path.join(BASE_DIR, "config", "runtime_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"保存运行时配置失败: {e}")

# 执行初始化
ensure_directories()
validate_config()