#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理模块，负责对接收到的图像进行预处理和后处理
"""

import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger("DNFAutoCloud")

def process_image(yolo_model, image):
    """
    处理图像并进行目标检测
    
    参数:
        yolo_model (YOLOModel): YOLO模型实例
        image (PIL.Image): 输入图像
        
    返回:
        list: 检测结果
    """
    try:
        # 图像预处理
        processed_image = preprocess_image(image)
        
        # 使用YOLO模型进行检测
        detections = yolo_model.detect(processed_image)
        
        # 后处理检测结果
        processed_detections = postprocess_detections(detections, image.size)
        
        return processed_detections
        
    except Exception as e:
        logger.error(f"图像处理出错: {e}")
        return []

def preprocess_image(image):
    """
    图像预处理
    
    参数:
        image (PIL.Image): 输入图像
        
    返回:
        PIL.Image: 预处理后的图像
    """
    try:
        # 图像增强
        image = image.convert("RGB")  # 确保图像是RGB模式
        
        # 调整亮度和对比度以便更好地检测
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # 增加对比度
        
        # 调整大小（可选，如果需要的话）
        # image = image.resize((640, 640), Image.LANCZOS)
        
        return image
        
    except Exception as e:
        logger.error(f"图像预处理出错: {e}")
        return image  # 返回原始图像

def postprocess_detections(detections, image_size):
    """
    后处理检测结果
    
    参数:
        detections (list): 检测结果
        image_size (tuple): 原始图像大小
        
    返回:
        list: 处理后的检测结果
    """
    try:
        # 过滤或合并重叠的检测框
        filtered_detections = []
        for det in detections:
            # 例如，过滤掉边缘的检测结果
            bbox = det['bbox']
            if is_valid_detection(bbox, image_size):
                # 添加相对位置信息
                det['center'] = [
                    (bbox[0] + bbox[2]) / 2,  # x中心
                    (bbox[1] + bbox[3]) / 2   # y中心
                ]
                det['size'] = [
                    bbox[2] - bbox[0],  # 宽度
                    bbox[3] - bbox[1]   # 高度
                ]
                det['relative_position'] = [
                    det['center'][0] / image_size[0],  # 相对x位置
                    det['center'][1] / image_size[1]   # 相对y位置
                ]
                
                filtered_detections.append(det)
        
        return filtered_detections
        
    except Exception as e:
        logger.error(f"检测结果后处理出错: {e}")
        return detections  # 返回原始检测结果

def is_valid_detection(bbox, image_size):
    """
    检查检测框是否有效
    
    参数:
        bbox (list): 边界框 [x1, y1, x2, y2]
        image_size (tuple): 图像大小 (width, height)
        
    返回:
        bool: 是否是有效的检测
    """
    # 检查框是否太小
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width < 10 or height < 10:
        return False
    
    # 检查框是否部分超出图像
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image_size[0] or bbox[3] > image_size[1]:
        return False
    
    return True