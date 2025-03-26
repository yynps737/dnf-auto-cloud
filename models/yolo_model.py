#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO模型封装，提供图像识别功能
"""

import os
import logging
import torch
import yaml
from PIL import Image
import numpy as np

from config.settings import MODEL

logger = logging.getLogger("DNFAutoCloud")

class YOLOModel:
    """YOLO模型封装类"""
    
    def __init__(self):
        """初始化YOLO模型"""
        self.device = MODEL["device"]
        self.conf_threshold = MODEL["conf_threshold"]
        self.iou_threshold = MODEL["iou_threshold"]
        
        logger.info(f"正在加载YOLO模型: {MODEL['name']}")
        logger.info(f"模型权重路径: {MODEL['weights']}")
        
        # 检查权重文件是否存在
        if not os.path.exists(MODEL["weights"]):
            raise FileNotFoundError(f"找不到模型权重文件: {MODEL['weights']}")
        
        # 加载模型
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                       path=MODEL["weights"], 
                                       device=self.device)
            
            # 设置模型参数
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            
            # 预热模型 (运行一个空白图像以初始化)
            dummy_img = torch.zeros((1, 3, 640, 640), device=self.device)
            self.model(dummy_img)
            
            logger.info("YOLO模型加载成功")
            
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {e}")
            raise
    
    def detect(self, image):
        """
        对图像进行目标检测
        
        参数:
            image (PIL.Image): 输入图像
            
        返回:
            list: 检测结果，每个结果包含边界框、类别和置信度
        """
        try:
            # 在GPU上进行推理
            with torch.no_grad():
                results = self.model(image)
            
            # 处理结果
            detections = []
            for pred in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = pred
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.model.names[int(cls)]
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"目标检测出错: {e}")
            return []
    
    def get_class_names(self):
        """获取类别名称列表"""
        return self.model.names