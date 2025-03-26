#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理模块，负责对接收到的图像进行预处理和后处理
优化版 - 增强图像处理性能和准确性
"""

import logging
import numpy as np
import cv2
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
        
        # 计算额外特征
        enhanced_detections = calculate_extra_features(processed_detections, image)
        
        return enhanced_detections
        
    except Exception as e:
        logger.error(f"图像处理出错: {e}")
        return []

def preprocess_image(image):
    """
    图像预处理 - 优化版
    
    参数:
        image (PIL.Image): 输入图像
        
    返回:
        PIL.Image: 预处理后的图像
    """
    try:
        # 确保图像是RGB模式
        image = image.convert("RGB")
        
        # 应用自适应直方图均衡化 (对于UI元素检测很有帮助)
        # 将PIL图像转换为numpy数组/OpenCV格式
        img_cv = np.array(image)
        img_cv = img_cv[:, :, ::-1].copy()  # RGB -> BGR
        
        # 转换为LAB色彩空间并对L通道应用CLAHE
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 合并处理后的通道
        limg = cv2.merge((cl, a, b))
        enhanced_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 将OpenCV图像转换回PIL格式
        enhanced = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
        
        # 适度锐化
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.3)
        
        # 稍微增加对比度
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"图像预处理出错: {e}")
        return image  # 返回原始图像

def postprocess_detections(detections, image_size):
    """
    后处理检测结果 - 优化版
    
    参数:
        detections (list): 检测结果
        image_size (tuple): 原始图像大小
        
    返回:
        list: 处理后的检测结果
    """
    try:
        # 过滤或合并重叠的检测框
        filtered_detections = []
        
        # 按类别和置信度排序
        detections.sort(key=lambda x: (x['class_name'], -x['confidence']))
        
        # 按类别分组
        class_groups = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        # 处理每个类别内的重叠框
        for class_name, dets in class_groups.items():
            # 使用非极大值抑制
            remaining = non_max_suppression(dets, iou_threshold=0.5)
            
            # 处理每个保留的检测
            for det in remaining:
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
                    # 添加一个唯一ID
                    det['id'] = f"{class_name}_{len(filtered_detections)}"
                    
                    filtered_detections.append(det)
        
        return filtered_detections
        
    except Exception as e:
        logger.error(f"检测结果后处理出错: {e}")
        return detections  # 返回原始检测结果

def non_max_suppression(detections, iou_threshold=0.5):
    """
    非极大值抑制实现
    
    参数:
        detections (list): 检测列表
        iou_threshold (float): IOU阈值
        
    返回:
        list: 抑制后的检测列表
    """
    # 如果列表为空，直接返回
    if not detections:
        return []
    
    # 获取所有边界框和置信度
    boxes = [d['bbox'] for d in detections]
    scores = [d['confidence'] for d in detections]
    
    # 初始化保留的框索引列表
    keep = []
    
    # 按置信度降序排序
    idxs = np.argsort(scores)[::-1]
    
    while len(idxs) > 0:
        # 取最高置信度的框
        current = idxs[0]
        keep.append(current)
        
        # 计算当前框与其他框的IoU
        ious = []
        current_box = boxes[current]
        
        for i in idxs[1:]:
            iou = calculate_iou(current_box, boxes[i])
            ious.append(iou)
        
        # 保留IoU小于阈值的框
        idxs = idxs[1:][np.array(ious) < iou_threshold]
    
    # 返回保留的检测
    return [detections[i] for i in keep]

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 两个边界框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # IoU
    iou = intersection / float(box1_area + box2_area - intersection)
    return iou

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
    if width < 5 or height < 5:
        return False
    
    # 检查框是否大部分超出图像
    if bbox[0] < -width/2 or bbox[1] < -height/2 or bbox[2] > image_size[0] + width/2 or bbox[3] > image_size[1] + height/2:
        return False
    
    return True

def calculate_extra_features(detections, image):
    """
    计算额外特征，例如血条/蓝条百分比
    
    参数:
        detections (list): 检测结果
        image (PIL.Image): 原始图像
        
    返回:
        list: 增强的检测结果
    """
    try:
        # 转换图像为numpy数组
        img_array = np.array(image)
        
        # 增强检测结果的额外信息
        for det in detections:
            # 处理血条
            if det["class_name"] == "hp_bar":
                det["percent"] = estimate_bar_percent(img_array, det["bbox"], "hp")
            
            # 处理蓝条
            elif det["class_name"] == "mp_bar":
                det["percent"] = estimate_bar_percent(img_array, det["bbox"], "mp")
            
            # 处理技能冷却
            elif det["class_name"] == "cooldown":
                # 如果有文本，尝试提取技能ID和剩余时间
                if "text" in det:
                    # 简单示例，真实情况可能需要OCR
                    det["skill_id"] = det.get("text", "unknown").split("_")[0]
                    det["remaining_time"] = 1.0  # 假设值
            
            # 处理物品
            elif det["class_name"] == "item":
                # 尝试识别物品稀有度（通过颜色）
                det["rarity"] = estimate_item_rarity(img_array, det["bbox"])
        
        return detections
        
    except Exception as e:
        logger.error(f"计算额外特征时出错: {e}")
        return detections

def estimate_bar_percent(img_array, bbox, bar_type):
    """
    估计血条/蓝条的百分比 - 优化版
    
    参数:
        img_array (numpy.ndarray): 图像数组
        bbox (list): 边界框 [x1, y1, x2, y2]
        bar_type (str): 条形类型 ("hp" 或 "mp")
        
    返回:
        float: 百分比值(0-100)
    """
    try:
        # 确保边界框在图像范围内
        x1, y1, x2, y2 = [int(v) for v in bbox]
        height, width = img_array.shape[:2]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        
        if x1 >= x2 or y1 >= y2:
            return 50.0  # 默认值
        
        # 提取条形区域
        bar_region = img_array[y1:y2, x1:x2]
        
        # 根据条形类型设置颜色阈值
        if bar_type == "hp":
            # 红色血条，查找红色像素
            lower_threshold = np.array([150, 0, 0])
            upper_threshold = np.array([255, 70, 70])
        elif bar_type == "mp":
            # 蓝色蓝条，查找蓝色像素
            lower_threshold = np.array([0, 0, 150])
            upper_threshold = np.array([70, 70, 255])
        else:
            return 50.0  # 默认值
        
        # 创建掩码
        mask = np.all((bar_region >= lower_threshold) & (bar_region <= upper_threshold), axis=2)
        
        # 计算比例（假设条形从左到右填充）
        bar_width = x2 - x1
        
        # 找出最右边的填充像素
        filled_width = 0
        for col in range(bar_width):
            if np.any(mask[:, col]):
                filled_width = col + 1
        
        # 计算百分比
        percent = (filled_width / bar_width) * 100
        
        return max(0, min(100, percent))  # 确保在0-100范围内
        
    except Exception as e:
        logger.error(f"估计条形百分比时出错: {e}")
        return 50.0  # 出错时返回默认值

def estimate_item_rarity(img_array, bbox):
    """
    估计物品稀有度（通过颜色）
    
    参数:
        img_array (numpy.ndarray): 图像数组
        bbox (list): 边界框 [x1, y1, x2, y2]
        
    返回:
        str: 稀有度描述
    """
    try:
        # 提取物品区域
        x1, y1, x2, y2 = [int(v) for v in bbox]
        height, width = img_array.shape[:2]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        
        # 提取边框区域（只取几个像素宽的边框）
        border_size = max(2, min(5, int((x2 - x1) * 0.1)))
        
        top_border = img_array[y1:y1+border_size, x1:x2]
        bottom_border = img_array[y2-border_size:y2, x1:x2]
        left_border = img_array[y1:y2, x1:x1+border_size]
        right_border = img_array[y1:y2, x2-border_size:x2]
        
        # 合并所有边框像素
        borders = np.vstack([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ])
        
        # 计算平均颜色
        avg_color = np.mean(borders, axis=0)
        
        # 根据颜色判断稀有度
        r, g, b = avg_color
        
        # 简单启发式规则
        if r > 200 and g < 100 and b < 100:
            return "legendary"  # 红色：传说
        elif r > 200 and g > 150 and b < 100:
            return "epic"  # 橙色：史诗
        elif r < 100 and g < 100 and b > 180:
            return "rare"  # 蓝色：稀有
        elif r < 100 and g > 180 and b < 100:
            return "uncommon"  # 绿色：优秀
        else:
            return "common"  # 白色/灰色：普通
        
    except Exception as e:
        logger.error(f"估计物品稀有度时出错: {e}")
        return "unknown"