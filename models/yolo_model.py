#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO模型封装，提供图像识别功能
优化版 - 增强性能和模型缓存管理
修复版 - 解决模型加载依赖问题
"""

import os
import sys  # 添加缺失的sys模块导入
import logging
import torch
import yaml
import time
import numpy as np
import subprocess
from PIL import Image
from collections import deque

from config.settings import MODEL, BASE_DIR

logger = logging.getLogger("DNFAutoCloud")

class YOLOModel:
    """YOLO模型封装类 - 优化版"""
    
    def __init__(self):
        """初始化YOLO模型"""
        self.device = MODEL.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = MODEL.get("conf_threshold", 0.5)
        self.iou_threshold = MODEL.get("iou_threshold", 0.45)
        
        # 性能监控
        self.inference_times = deque(maxlen=100)
        self.batch_size = MODEL.get("batch_size", 1)
        self.use_half_precision = MODEL.get("half_precision", True) and self.device == "cuda"
        
        logger.info(f"正在加载YOLO模型: {MODEL.get('name', 'unknown')}")
        logger.info(f"模型权重路径: {MODEL.get('weights', 'unknown')}")
        logger.info(f"设备: {self.device}, 半精度: {self.use_half_precision}")
        
        # 检查权重文件是否存在
        weights_path = MODEL.get("weights", "")
        if not os.path.exists(weights_path):
            # 尝试从备选路径加载
            alt_paths = [
                os.path.join(BASE_DIR, "models", "weights", "best.pt"),
                os.path.join(BASE_DIR, "models", "best.pt"),
                os.path.join(BASE_DIR, "yolov5", "runs", "train", "exp", "weights", "best.pt")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    weights_path = path
                    logger.info(f"使用备选模型权重: {weights_path}")
                    break
            
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"找不到模型权重文件: {weights_path}")
        
        # 加载模型
        try:
            if MODEL.get("engine", "") == "onnx":
                self._load_onnx_model()
            else:
                # 尝试不同的加载方法
                try:
                    self._load_yolov5_custom()
                except Exception as e:
                    logger.warning(f"使用torch.hub加载YOLOv5模型失败: {e}")
                    logger.info("尝试备选方法加载模型...")
                    self._load_yolov5_manually()
            
            logger.info("YOLO模型加载成功")
            
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {e}")
            raise
    
    def _load_yolov5_custom(self):
        """使用torch.hub加载YOLOv5模型"""
        try:
            # 尝试从本地加载YOLOv5
            yolov5_dir = os.path.join(BASE_DIR, "tools", "yolov5")
            if os.path.exists(yolov5_dir):
                logger.info(f"从本地目录加载YOLOv5: {yolov5_dir}")
                sys.path.insert(0, yolov5_dir)
                
                try:
                    # 尝试直接导入YOLOv5模块
                    from models.common import DetectMultiBackend
                    from utils.torch_utils import select_device
                    from utils.general import check_img_size, non_max_suppression, scale_coords
                    
                    # 加载模型
                    self.model = DetectMultiBackend(MODEL.get("weights"), device=self.device)
                    self.stride = self.model.stride
                    self.names = self.model.names
                    
                    # 设置参数
                    self.imgsz = check_img_size((640, 640), s=self.stride)
                    
                    # 使用半精度
                    if self.use_half_precision:
                        self.model.half()
                    
                    # 预热模型
                    self._warmup_model()
                    
                    return
                except ImportError as e:
                    logger.warning(f"无法直接导入YOLOv5模块: {e}")
            
            # 尝试从torch.hub加载
            logger.info("尝试从torch.hub加载YOLOv5模型")
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                'custom', 
                path=MODEL.get("weights"),
                device=self.device,
                force_reload=True
            )
            
            # 设置模型参数
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            
            # 使用半精度
            if self.use_half_precision:
                self.model.half()
            
            # 预热模型
            self._warmup_model()
            
        except Exception as e:
            logger.error(f"加载YOLOv5模型失败: {e}")
            raise
    
    def _load_yolov5_manually(self):
        """手动加载YOLOv5模型（不依赖torch.hub）"""
        try:
            # 检查YOLOv5目录是否存在，不存在则克隆
            yolov5_dir = os.path.join(BASE_DIR, "tools", "yolov5")
            if not os.path.exists(yolov5_dir):
                logger.info("YOLOv5目录不存在，正在克隆仓库...")
                os.makedirs(os.path.dirname(yolov5_dir), exist_ok=True)
                
                # 克隆YOLOv5仓库
                subprocess.run(
                    ["git", "clone", "https://github.com/ultralytics/yolov5.git", yolov5_dir],
                    check=True
                )
            
            # 添加YOLOv5目录到系统路径
            sys.path.insert(0, yolov5_dir)
            
            try:
                # 尝试导入YOLOv5模块
                from models.common import DetectMultiBackend
                from utils.torch_utils import select_device
                from utils.general import check_img_size, non_max_suppression, scale_coords
                
                # 加载模型
                self.model = DetectMultiBackend(MODEL.get("weights"), device=self.device)
                self.stride = self.model.stride
                self.names = self.model.names
                
                # 设置参数
                self.imgsz = check_img_size((640, 640), s=self.stride)
                
                # 使用半精度
                if self.use_half_precision:
                    self.model.half()
                
                # 保存必要的函数
                self.non_max_suppression = non_max_suppression
                self.scale_coords = scale_coords
                
                # 预热模型
                dummy_img = torch.zeros((1, 3, self.imgsz[0], self.imgsz[1]), device=self.device)
                if self.use_half_precision:
                    dummy_img = dummy_img.half()
                
                with torch.no_grad():
                    for _ in range(2):
                        self.model(dummy_img)
                
                logger.info("手动加载YOLOv5模型成功")
                
            except ImportError as e:
                logger.error(f"导入YOLOv5模块失败: {e}")
                # 尝试使用更简单的模型
                self._load_fallback_model()
        
        except Exception as e:
            logger.error(f"手动加载YOLOv5模型失败: {e}")
            # 尝试使用更简单的模型
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """加载备用模型（使用PyTorch内置模型）"""
        logger.info("尝试加载备用模型 (PyTorch YOLO)")
        
        try:
            # 使用PyTorch的预训练模型
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # 备用模型的类别
            self.names = [
                'background', 'monster', 'boss', 'door', 'item', 'npc', 'player', 
                'hp_bar', 'mp_bar', 'skill_ready', 'cooldown'
            ]
            
            # 标记使用备用模型
            self.using_fallback = True
            
            logger.info("备用模型加载成功")
            
        except Exception as e:
            logger.error(f"加载备用模型失败: {e}")
            raise
    
    def _load_onnx_model(self):
        """加载ONNX版YOLO模型"""
        try:
            import onnxruntime as ort
            
            # 设置ONNX运行时参数
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # 检查ONNX模型文件
            onnx_path = MODEL.get("weights", "")
            if not onnx_path.endswith('.onnx'):
                onnx_path = onnx_path + '.onnx'
            
            if not os.path.exists(onnx_path):
                logger.warning(f"找不到ONNX模型: {onnx_path}")
                logger.info("尝试导出PyTorch模型到ONNX格式...")
                
                # 尝试加载PyTorch模型并导出为ONNX
                self._load_yolov5_custom()
                self._export_to_onnx(onnx_path)
            
            # 创建ONNX会话
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 获取模型输入输出名称
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_names = [output.name for output in self.onnx_session.get_outputs()]
            
            # 加载类别名称
            if os.path.exists(MODEL.get("class_names", "")):
                with open(MODEL.get("class_names"), "r") as f:
                    self.class_names = yaml.safe_load(f)
            else:
                self.class_names = ["object"]
            
            # 标记使用ONNX
            self.using_onnx = True
            
            # 预热模型
            self._warmup_onnx_model()
            
        except ImportError:
            logger.error("缺少ONNX运行时依赖，请安装onnxruntime或onnxruntime-gpu")
            logger.info("尝试使用PyTorch模型代替...")
            self._load_yolov5_custom()
        except Exception as e:
            logger.error(f"加载ONNX模型失败: {e}")
            logger.info("尝试使用PyTorch模型代替...")
            self._load_yolov5_custom()
    
    def _export_to_onnx(self, onnx_path):
        """将PyTorch模型导出为ONNX格式"""
        try:
            import torch.onnx
            
            # 准备导出
            dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
            if self.use_half_precision:
                dummy_input = dummy_input.half()
            
            # 导出ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                verbose=False,
                opset_version=12,
                input_names=['images'],
                output_names=['output']
            )
            
            logger.info(f"PyTorch模型成功导出为ONNX格式: {onnx_path}")
            
        except Exception as e:
            logger.error(f"导出ONNX模型失败: {e}")
            raise
    
    def _warmup_model(self):
        """预热模型（运行空白图像以初始化）"""
        dummy_img = torch.zeros((1, 3, 640, 640), device=self.device)
        if self.use_half_precision:
            dummy_img = dummy_img.half()
        
        # 进行多次推理预热
        with torch.no_grad():
            for _ in range(2):
                self.model(dummy_img)
        
        logger.info("模型预热完成")
    
    def _warmup_onnx_model(self):
        """预热ONNX模型"""
        dummy_img = np.zeros((1, 3, 640, 640), dtype=np.float32)
        
        # 进行多次推理预热
        for _ in range(2):
            self.onnx_session.run(self.output_names, {self.input_name: dummy_img})
        
        logger.info("ONNX模型预热完成")
    
    def detect(self, image):
        """
        对图像进行目标检测
        
        参数:
            image (PIL.Image): 输入图像
            
        返回:
            list: 检测结果，每个结果包含边界框、类别和置信度
        """
        try:
            start_time = time.time()
            
            # 根据模型类型选择检测方法
            if hasattr(self, 'using_onnx') and self.using_onnx:
                detections = self._detect_onnx(image)
            elif hasattr(self, 'using_fallback') and self.using_fallback:
                detections = self._detect_fallback(image)
            else:
                detections = self._detect_pytorch(image)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 计算平均推理时间
            avg_time = sum(self.inference_times) / len(self.inference_times)
            
            if len(self.inference_times) % 10 == 0:
                logger.debug(f"平均推理时间: {avg_time:.3f}秒, 当前: {inference_time:.3f}秒")
            
            return detections
            
        except Exception as e:
            logger.error(f"目标检测出错: {e}")
            return []
    
    def _detect_pytorch(self, image):
        """使用PyTorch模型进行检测"""
        # 在GPU上进行推理
        with torch.no_grad():
            # 处理不同的模型接口
            if hasattr(self, 'stride') and hasattr(self, 'names'):
                # 自定义加载的YOLOv5
                # 转换图像
                img = self._prepare_image_custom(image)
                
                # 推理
                output = self.model(img)
                
                # 处理输出
                pred = self.non_max_suppression(output[0], self.conf_threshold, self.iou_threshold)
                
                # 解析结果
                detections = []
                for det in pred[0]:
                    x1, y1, x2, y2, conf, cls = det
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.names[int(cls)]
                    })
            else:
                # 标准torch.hub加载的模型
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
    
    def _detect_fallback(self, image):
        """使用备用模型进行检测"""
        # 预处理图像
        img = self._prepare_image_pytorch(image)
        
        # 进行推理
        with torch.no_grad():
            predictions = self.model(img)
        
        # 处理结果
        detections = []
        for i, prediction in enumerate(predictions[0]['boxes']):
            score = predictions[0]['scores'][i].item()
            if score > self.conf_threshold:
                x1, y1, x2, y2 = prediction.tolist()
                class_id = predictions[0]['labels'][i].item()
                
                # 映射torchvision的COCO类别到我们的类别
                class_name = self.names[min(class_id, len(self.names) - 1)]
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def _detect_onnx(self, image):
        """使用ONNX模型进行检测"""
        # 预处理图像
        input_tensor = self._prepare_image_onnx(image)
        
        # 运行推理
        outputs = self.onnx_session.run(self.output_names, {self.input_name: input_tensor})
        
        # 解析输出（根据ONNX模型的输出格式可能需要调整）
        # 假设输出格式为 [batch_id, x1, y1, x2, y2, conf, class_id]
        predictions = outputs[0]
        
        # 过滤低置信度预测
        mask = predictions[:, 5] > self.conf_threshold
        filtered_preds = predictions[mask]
        
        # 组织检测结果
        detections = []
        for pred in filtered_preds:
            x1, y1, x2, y2, conf, cls = pred[1:7]
            cls_id = int(cls)
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': cls_id,
                'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else "unknown"
            })
        
        return detections
    
    def _prepare_image_custom(self, image):
        """为自定义加载的YOLOv5模型准备图像"""
        # 转换PIL图像为numpy数组
        img = np.array(image)
        
        # 调整大小
        img = self._letterbox(img, new_shape=self.imgsz)[0]
        
        # BGR转RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        # 转换为PyTorch张量
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.use_half_precision else img.float()
        img /= 255.0
        
        # 增加批次维度
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img
    
    def _prepare_image_pytorch(self, image):
        """为PyTorch模型准备图像"""
        # 转换PIL图像为PyTorch张量
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        img = transform(image).unsqueeze(0).to(self.device)
        
        return img
    
    def _prepare_image_onnx(self, image):
        """为ONNX模型准备图像"""
        # 调整图像大小
        img_size = MODEL.get("img_size", 640)
        image = image.resize((img_size, img_size), Image.LANCZOS)
        
        # 转换为numpy数组
        img = np.array(image).astype(np.float32) / 255.0
        
        # 从HWC转换为CHW格式
        img = img.transpose(2, 0, 1)
        
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """调整图像大小并填充（YOLOv5风格）"""
        shape = img.shape[:2]  # 当前形状 [高, 宽]
        
        # 缩放比例 (新 / 旧)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        # 平均分配填充
        dw /= 2
        dh /= 2
        
        # 调整大小
        if shape[::-1] != new_unpad:
            import cv2
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 添加边框
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)
    
    def get_class_names(self):
        """获取类别名称列表"""
        if hasattr(self, 'using_onnx') and self.using_onnx:
            return self.class_names
        elif hasattr(self, 'using_fallback') and self.using_fallback:
            return self.names
        elif hasattr(self, 'names'):
            return self.names
        else:
            return self.model.names
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return {"average_time": 0, "min_time": 0, "max_time": 0, "count": 0, "fps": 0}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        
        # 修复fps计算逻辑
        if avg_time > 0:
            fps = 1.0 / avg_time
        else:
            fps = 0
            
        return {
            "average_time": avg_time,
            "min_time": min(self.inference_times),
            "max_time": max(self.inference_times),
            "count": len(self.inference_times),
            "fps": fps
        }