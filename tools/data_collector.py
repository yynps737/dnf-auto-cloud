#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据收集工具 - 适用于Linux服务器环境
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
import threading
import queue

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging_utils import setup_logging

# 设置日志
setup_logging(debug=True)
logger = logging.getLogger("DataCollector")

class DNFDataCollector:
    """DNF数据收集器类 - 服务器版本"""
    
    def __init__(self, output_dir="data/training", interval=1.0):
        """
        初始化数据收集器
        
        参数:
            output_dir (str): 输出目录
            interval (float): 截图间隔（秒）
        """
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.running = False
        self.total_captured = 0
        
        # 创建输出目录
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 类别定义
        self.classes = [
            "monster", "boss", "door", "item", "npc", "player", 
            "hp_bar", "mp_bar", "skill_ready", "cooldown"
        ]
        
        logger.info(f"数据收集器初始化完成，输出目录: {self.output_dir}")
        logger.info(f"类别定义: {self.classes}")
    
    def create_sample_images(self, count=10):
        """
        创建示例图像用于测试模型训练流程
        
        参数:
            count (int): 要创建的示例图像数量
        """
        logger.info(f"正在创建 {count} 个示例图像...")
        
        # 创建训练和验证子目录
        train_dir = self.images_dir / "train"
        val_dir = self.images_dir / "val"
        test_dir = self.images_dir / "test"
        
        train_labels_dir = self.labels_dir / "train"
        val_labels_dir = self.labels_dir / "val"
        test_labels_dir = self.labels_dir / "test"
        
        for directory in [train_dir, val_dir, test_dir, 
                          train_labels_dir, val_labels_dir, test_labels_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 创建示例图像
        for i in range(count):
            # 决定放入哪个目录
            if i < count * 0.7:  # 70% 训练集
                img_dir = train_dir
                label_dir = train_labels_dir
            elif i < count * 0.9:  # 20% 验证集
                img_dir = val_dir
                label_dir = val_labels_dir
            else:  # 10% 测试集
                img_dir = test_dir
                label_dir = test_labels_dir
            
            # 创建一个640x480的黑色背景图像
            img = Image.new('RGB', (640, 480), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # 添加一些简单的形状作为示例对象
            labels = []
            
            # 添加1-3个随机矩形，模拟怪物
            monster_count = random.randint(1, 3)
            for j in range(monster_count):
                # 随机位置和大小
                x1 = random.randint(50, 550)
                y1 = random.randint(50, 400)
                width = random.randint(30, 80)
                height = random.randint(30, 80)
                x2 = x1 + width
                y2 = y1 + height
                
                # 随机颜色
                color = (random.randint(100, 255), 
                        random.randint(100, 255),
                        random.randint(100, 255))
                
                # 绘制矩形
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # 添加类别标签 (YOLO格式: <class_id> <x_center> <y_center> <width> <height>)
                # 所有值都是相对于图像大小的比例
                class_id = 0  # 0 = monster
                x_center = (x1 + x2) / 2 / 640
                y_center = (y1 + y2) / 2 / 480
                w = (x2 - x1) / 640
                h = (y2 - y1) / 480
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # 有10%的概率添加一个boss
            if random.random() < 0.1:
                x1 = random.randint(100, 500)
                y1 = random.randint(100, 350)
                width = random.randint(60, 100)
                height = random.randint(60, 100)
                x2 = x1 + width
                y2 = y1 + height
                
                color = (255, 50, 50)  # 红色
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                class_id = 1  # 1 = boss
                x_center = (x1 + x2) / 2 / 640
                y_center = (y1 + y2) / 2 / 480
                w = (x2 - x1) / 640
                h = (y2 - y1) / 480
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # 添加门或物品
            if random.random() < 0.5:
                # 门
                x1 = random.randint(250, 350)
                y1 = random.randint(200, 300)
                width = random.randint(40, 60)
                height = random.randint(60, 80)
                x2 = x1 + width
                y2 = y1 + height
                
                color = (50, 200, 50)  # 绿色
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                class_id = 2  # 2 = door
                x_center = (x1 + x2) / 2 / 640
                y_center = (y1 + y2) / 2 / 480
                w = (x2 - x1) / 640
                h = (y2 - y1) / 480
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            else:
                # 物品
                for j in range(random.randint(0, 2)):
                    x1 = random.randint(100, 500)
                    y1 = random.randint(100, 400)
                    size = random.randint(10, 25)
                    x2 = x1 + size
                    y2 = y1 + size
                    
                    color = (200, 200, 50)  # 黄色
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    class_id = 3  # 3 = item
                    x_center = (x1 + x2) / 2 / 640
                    y_center = (y1 + y2) / 2 / 480
                    w = (x2 - x1) / 640
                    h = (y2 - y1) / 480
                    
                    labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i:04d}"
            filename = f"sample_{timestamp}.jpg"
            filepath = img_dir / filename
            img.save(filepath)
            
            # 保存标签
            label_filepath = label_dir / f"{Path(filename).stem}.txt"
            with open(label_filepath, "w") as f:
                f.write("\n".join(labels))
            
            if i % 10 == 0 and i > 0:
                logger.info(f"已创建 {i} 个示例图像")
        
        logger.info(f"示例图像创建完成，共 {count} 个")
        
        # 创建数据集配置文件
        self.create_dataset_config()
        
        return count
    
    def create_dataset_config(self):
        """创建YOLO数据集配置文件（修复版）"""
        data_yaml_path = self.output_dir / "data.yaml"
        
        # 使用绝对路径
        abs_data_dir = self.output_dir.absolute()
        
        # 确保路径正确，YOLOv5期望的格式
        data_config = {
            "path": str(abs_data_dir),
            "train": str((self.images_dir / "train").absolute()),
            "val": str((self.images_dir / "val").absolute()),
            "test": str((self.images_dir / "test").absolute()),
            "nc": len(self.classes),
            "names": self.classes
        }
        
        # 保存配置
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"数据集配置文件已创建: {data_yaml_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DNF数据收集工具 - 服务器版本")
    parser.add_argument("--output", type=str, default="data/training", help="输出目录")
    parser.add_argument("--interval", type=float, default=1.0, help="截图间隔（秒）")
    parser.add_argument("--duration", type=float, default=None, help="持续时间（秒），默认无限")
    parser.add_argument("--create-samples", type=int, default=100, help="创建示例图像数量（用于测试训练流程）")
    args = parser.parse_args()
    
    # 创建数据收集器
    collector = DNFDataCollector(
        output_dir=args.output,
        interval=args.interval
    )
    
    # 创建示例图像
    if args.create_samples > 0:
        collector.create_sample_images(count=args.create_samples)

if __name__ == "__main__":
    main()