#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO模型实现，负责目标检测和识别
支持多种YOLO版本：YOLOv5、YOLOv8
优化版 - 支持GPU加速和模型优化
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
import torch
import cv2
from PIL import Image
import statistics
from collections import deque

from config.settings import MODEL

logger = logging.getLogger("DNFAutoCloud")

class YOLOModel:
    """YOLO模型类 - 优化版"""
    
    def __init__(self):
        """初始化YOLO模型"""
        self.model = None
        self.device = MODEL.get("device", "cpu")
        self.conf_threshold = MODEL.get("conf_threshold", 0.5)
        self.iou_threshold = MODEL.get("iou_threshold", 0.45)
        self.img_size = MODEL.get("img_size", 640)
        self.half_precision = MODEL.get("half_precision", False)
        self.class_names = self._load_class_names()
        
        # 性能追踪
        self.inference_times = deque(maxlen=100)
        
        # 初始化模型
        self._initialize_model()
    
    def _load_class_names(self):
        """加载类别名称"""
        try:
            class_names_path = MODEL.get("class_names", "")
            if not class_names_path or not os.path.exists(class_names_path):
                logger.warning(f"类别名称文件不存在: {class_names_path}，使用默认类别")
                return [
                    "monster", "boss", "door", "item", "npc", "player", 
                    "hp_bar", "mp_bar", "skill_ready", "cooldown"
                ]
            
            # 尝试从YAML加载
            try:
                import yaml
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if isinstance(data, dict) and "names" in data:
                    return data["names"]
                elif "nc" in data and isinstance(data.get("names", []), list):
                    return data["names"]
            except:
                # 如果YAML加载失败，尝试直接读取为文本
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    class_names = [line.strip() for line in f.readlines() if line.strip()]
                return class_names
                
        except Exception as e:
            logger.error(f"加载类别名称时出错: {e}")
            return [f"class_{i}" for i in range(10)]  # 默认类别名
    
    def _initialize_model(self):
        """初始化YOLO模型"""
        try:
            # 获取模型路径
            weights_path = MODEL.get("weights", "")
            if not weights_path:
                raise ValueError("未指定模型权重路径")
            
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
            
            logger.info(f"正在加载YOLO模型: {weights_path}")
            
            # 确定模型引擎类型
            engine = MODEL.get("engine", "pytorch").lower()
            
            if engine == "onnx":
                # 使用ONNX模型
                if not weights_path.endswith(".onnx"):
                    weights_path = weights_path + ".onnx"
                
                self._initialize_onnx_model(weights_path)
            else:
                # 默认使用PyTorch模型
                self._initialize_pytorch_model(weights_path)
            
            logger.info(f"YOLO模型加载成功，运行于 {self.device} 设备")
            
        except Exception as e:
            logger.error(f"初始化YOLO模型失败: {e}")
            raise
    
    def _initialize_pytorch_model(self, weights_path):
        """初始化PyTorch YOLO模型"""
        try:
            # 检查设备可用性
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA不可用，切换到CPU模式")
                self.device = "cpu"
                self.half_precision = False
            
            # 设置设备
            device = torch.device(self.device)
            
            # 确定YOLO版本并加载
            model_name = MODEL.get("name", "").lower()
            
            if "yolov8" in model_name or weights_path.endswith("v8.pt"):
                # YOLOv8
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(weights_path)
                    self.model_type = "yolov8"
                    logger.info("已加载YOLOv8模型")
                except ImportError:
                    logger.error("无法导入ultralytics，请安装: pip install ultralytics")
                    raise
            else:
                # YOLOv5（默认）
                try:
                    sys.path.append(os.path.join(os.path.dirname(__file__), "../tools/yolov5"))
                    import torch
                    
                    # 加载模型
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                              path=weights_path, device=device)
                    
                    # 设置参数
                    self.model.conf = self.conf_threshold
                    self.model.iou = self.iou_threshold
                    self.model.classes = None  # 检测所有类别
                    self.model.max_det = 100   # 最大检测数量
                    
                    # 如果启用半精度且支持
                    if self.half_precision and self.device != "cpu":
                        self.model.half()
                    
                    self.model_type = "yolov5"
                    logger.info("已加载YOLOv5模型")
                    
                except ImportError:
                    logger.error("无法导入torch或YOLOv5，请确保已正确安装")
                    raise
                except Exception as e:
                    logger.error(f"加载YOLOv5模型时出错: {e}")
                    raise
            
            # 预热模型
            self._warmup_model()
            
        except Exception as e:
            logger.error(f"初始化PyTorch模型失败: {e}")
            raise
    
    def _initialize_onnx_model(self, weights_path):
        """初始化ONNX YOLO模型"""
        try:
            import onnxruntime as ort
            
            # 设置ONNX运行时选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 选择运行设备
            if self.device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                self.device = "cpu"
            
            # 创建ONNX会话
            self.model = ort.InferenceSession(weights_path, sess_options=sess_options, 
                                            providers=providers)
            
            # 获取模型输入输出信息
            self.input_names = [input.name for input in self.model.get_inputs()]
            self.output_names = [output.name for output in self.model.get_outputs()]
            
            self.model_type = "onnx"
            logger.info(f"已加载ONNX模型，提供者: {self.model.get_providers()}")
            
            # 预热模型
            dummy_input = np.random.rand(1, 3, self.img_size, self.img_size).astype(np.float32)
            input_feed = {self.input_names[0]: dummy_input}
            self.model.run(self.output_names, input_feed)
            
        except ImportError:
            logger.error("无法导入onnxruntime，请安装: pip install onnxruntime-gpu 或 onnxruntime")
            raise
        except Exception as e:
            logger.error(f"初始化ONNX模型失败: {e}")
            raise
    
    def _warmup_model(self):
        """预热模型，避免首次推理延迟"""
        try:
            logger.info("预热模型...")
            
            if self.model_type == "yolov8":
                # YOLOv8预热
                dummy_image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
                self.model(dummy_image, verbose=False)
                
            elif self.model_type == "yolov5":
                # YOLOv5预热
                dummy_image = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
                if self.half_precision and self.device != "cpu":
                    dummy_image = dummy_image.half()
                
                self.model(dummy_image)
            
            logger.info("模型预热完成")
            
        except Exception as e:
            logger.warning(f"模型预热出错: {e}")
    
    def detect(self, image):
        """
        执行目标检测
        
        参数:
            image (PIL.Image): 输入图像
            
        返回:
            list: 检测结果列表
        """
        try:
            start_time = time.time()
            
            # 检查模型是否已初始化
            if self.model is None:
                logger.error("模型未初始化")
                return []
            
            # 选择相应的检测方法
            if self.model_type == "yolov8":
                detections = self._detect_yolov8(image)
            elif self.model_type == "yolov5":
                detections = self._detect_yolov5(image)
            elif self.model_type == "onnx":
                detections = self._detect_onnx(image)
            else:
                logger.error(f"不支持的模型类型: {self.model_type}")
                return []
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return detections
            
        except Exception as e:
            logger.error(f"目标检测出错: {e}")
            return []
    
    def _detect_yolov5(self, image):
        """使用YOLOv5模型进行检测"""
        # 转换PIL图像为所需格式
        if isinstance(image, Image.Image):
            # PIL图像直接传给模型
            results = self.model(image, size=self.img_size)
        else:
            # 转换numpy数组为PIL图像
            image = Image.fromarray(np.array(image))
            results = self.model(image, size=self.img_size)
        
        # 提取结果
        predictions = results.xyxy[0].cpu().numpy()  # xyxy格式，(n, 6) - x1, y1, x2, y2, conf, cls
        
        # 转换为标准格式
        detections = []
        for pred in predictions:
            x1, y1, x2, y2, conf, cls_id = pred
            cls_id = int(cls_id)
            
            # 确保类别ID在范围内
            if cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                class_name = f"class_{cls_id}"
            
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "class_id": cls_id,
                "class_name": class_name
            })
        
        return detections
    
    def _detect_yolov8(self, image):
        """使用YOLOv8模型进行检测"""
        # 转换PIL图像为所需格式
        if isinstance(image, Image.Image):
            # 转换为numpy数组
            image_np = np.array(image)
        else:
            image_np = np.array(image)
        
        # 执行推理
        results = self.model(
            image_np, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # 提取结果
        detections = []
        
        for result in results:
            # 获取边界框
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # 提取坐标，conf和类别
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls_id = int(box.cls)
                
                # 确保类别ID在范围内
                if cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": class_name
                })
        
        return detections
    
    def _detect_onnx(self, image):
        """使用ONNX模型进行检测"""
        # 准备输入图像
        if isinstance(image, Image.Image):
            # 转换PIL图像为numpy数组
            img = np.array(image.resize((self.img_size, self.img_size)))
        else:
            # 转换任何其他格式为numpy数组
            img = cv2.resize(np.array(image), (self.img_size, self.img_size))
        
        # 图像预处理
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # 添加batch维度
        img = img / 255.0  # 归一化到[0,1]
        img = img.astype(np.float32)  # 转换为float32
        
        # 执行推理
        input_feed = {self.input_names[0]: img}
        outputs = self.model.run(self.output_names, input_feed)
        
        # 后处理
        # 注意：实际后处理可能因模型而异
        # 这里假设输出为[batch, num_detections, 6]格式，其中6为[x1, y1, x2, y2, conf, cls]
        predictions = outputs[0]  # 假设第一个输出是检测结果
        
        # 转换为标准格式
        detections = []
        
        # 提取有效检测
        valid_preds = predictions[0]  # 取第一个batch
        
        for pred in valid_preds:
            if len(pred) >= 6:  # 检查是否有6个元素
                x1, y1, x2, y2, conf, cls_id = pred[:6]
                
                # 过滤低置信度检测
                if conf < self.conf_threshold:
                    continue
                
                cls_id = int(cls_id)
                
                # 确保类别ID在范围内
                if cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                # 将坐标从[0,1]缩放到原始图像尺寸
                if max(x1, y1, x2, y2) <= 1.0:
                    # 坐标是归一化的，需要缩放到原始尺寸
                    if hasattr(image, 'size'):
                        orig_w, orig_h = image.size
                    else:
                        orig_h, orig_w = image.shape[:2]
                    
                    x1 *= orig_w
                    x2 *= orig_w
                    y1 *= orig_h
                    y2 *= orig_h
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": cls_id,
                    "class_name": class_name
                })
        
        return detections
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return {
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "fps": 0
            }
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return {
            "average_time": avg_time,
            "min_time": min(self.inference_times),
            "max_time": max(self.inference_times),
            "median_time": statistics.median(self.inference_times),
            "fps": 1.0 / avg_time if avg_time > 0 else 0
        }

if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 初始化模型
        model = YOLOModel()
        
        # 加载测试图像
        test_image_path = "data/training/images/test/sample_1.jpg"
        if os.path.exists(test_image_path):
            image = Image.open(test_image_path)
            
            # 检测
            detections = model.detect(image)
            
            # 打印结果
            print(f"检测到 {len(detections)} 个目标:")
            for det in detections:
                print(f"类别: {det['class_name']}, 置信度: {det['confidence']:.2f}, 边界框: {det['bbox']}")
            
            # 打印性能
            perf = model.get_performance_stats()
            print(f"模型性能: {perf['average_time']*1000:.2f}ms, {perf['fps']:.2f} FPS")
        else:
            print(f"测试图像不存在: {test_image_path}")
    
    except Exception as e:
        print(f"测试失败: {e}")