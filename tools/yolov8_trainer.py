#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8模型训练工具 - 用于训练YOLO模型识别DNF游戏中的对象
"""

import os
import sys
import yaml
import argparse
import logging
import shutil
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging_utils import setup_logging

# 设置日志
setup_logging(debug=True)
logger = logging.getLogger("ModelTrainer")

class YOLOv8Trainer:
    """YOLOv8模型训练器类"""
    
    def __init__(self, config_path="config/model_config.yaml"):
        """
        初始化训练器
        
        参数:
            config_path (str): 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # 检查配置
        self.validate_config()
        
        logger.info(f"模型训练器初始化完成，配置文件: {self.config_path}")
    
    def load_config(self):
        """
        加载配置文件
        
        返回:
            dict: 配置信息
        """
        try:
            if not self.config_path.exists():
                logger.error(f"配置文件不存在: {self.config_path}")
                return {}
            
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            logger.info(f"已加载配置: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件时出错: {e}")
            return {}
    
    def validate_config(self):
        """验证配置"""
        required_keys = ["data_dir", "weights_dir", "batch_size", "epochs", "img_size"]
        
        for key in required_keys:
            if key not in self.config:
                logger.warning(f"配置中缺少 {key}")
    
    def prepare_data(self):
        """准备训练数据"""
        data_dir = Path(self.config.get("data_dir", "data/training"))
        
        if not data_dir.exists():
            logger.error(f"数据目录不存在: {data_dir}")
            return False
        
        # 验证数据目录结构
        images_dir = data_dir / "images"
        train_dir = images_dir / "train"
        val_dir = images_dir / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            logger.warning(f"训练/验证目录不存在: {train_dir} / {val_dir}")
            logger.warning("请先运行 data_collector.py 生成训练数据")
            return False
        
        # 统计图像数量
        train_images = list(train_dir.glob("*.jpg"))
        val_images = list(val_dir.glob("*.jpg"))
        
        # 验证YAML配置
        data_yaml_path = data_dir / "data.yaml"
        if not data_yaml_path.exists():
            logger.warning(f"数据配置文件不存在: {data_yaml_path}")
            self.create_dataset_config(data_dir)
        
        logger.info(f"训练图像: {len(train_images)}, 验证图像: {len(val_images)}")
        
        return len(train_images) > 0 and len(val_images) > 0
    
    def create_dataset_config(self, data_dir):
        """创建YOLO数据集配置文件"""
        data_yaml_path = data_dir / "data.yaml"
        
        # 使用绝对路径
        abs_data_dir = data_dir.absolute()
        
        # 确保路径正确
        data_config = {
            "path": str(abs_data_dir),
            "train": "images/train",  # YOLOv8使用相对路径
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.config.get("classes", [])),
            "names": self.config.get("classes", [])
        }
        
        # 保存配置
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"数据集配置文件已创建: {data_yaml_path}")
    
    def train_model(self):
        """训练模型"""
        try:
            # 导入ultralytics
            from ultralytics import YOLO
            
            # 准备数据
            if not self.prepare_data():
                return False
            
            # 获取训练参数
            data_dir = Path(self.config.get("data_dir", "data/training"))
            data_yaml_path = data_dir / "data.yaml"
            
            weights_dir = Path(self.config.get("weights_dir", "models/weights"))
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            batch_size = self.config.get("batch_size", 16)
            epochs = self.config.get("epochs", 100)
            img_size = self.config.get("img_size", 640)
            device = self.config.get("device", "0")
            
            # 选择预训练权重
            pretrained = self.config.get("pretrained", "yolov8m.pt")
            
            # 加载模型
            model = YOLO(pretrained)
            
            # 训练模型
            logger.info(f"开始训练模型，使用数据: {data_yaml_path}")
            
            results = model.train(
                data=str(data_yaml_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                project=str(weights_dir.parent),
                name=weights_dir.name,
                exist_ok=True,
                pretrained=True,
                verbose=True
            )
            
            # 复制最终模型 - 修复后的代码
            try:
                # 尝试从results获取best路径
                if results and hasattr(results, 'best') and results.best:
                    best_model_path = Path(results.best)
                    if best_model_path.exists():
                        target_path = weights_dir / "dnf_yolo8m.pt"
                        shutil.copy(best_model_path, target_path)
                        logger.info(f"已复制最佳模型到: {target_path}")
                    else:
                        logger.warning(f"最佳模型路径不存在: {best_model_path}")
                else:
                    # 尝试查找最佳模型
                    logger.info("无法从结果获取最佳模型路径，尝试查找训练输出目录中的模型文件")
                    
                    # 检查不同可能的目录结构
                    possible_dirs = [
                        weights_dir / 'weights',  # 常见的weights子目录
                        weights_dir / 'train' / 'weights',  # ultralytics常用的结构
                        weights_dir  # 直接在weights_dir下
                    ]
                    
                    found = False
                    for run_dir in possible_dirs:
                        if run_dir.exists():
                            # 寻找best.pt或最后一个epoch的pt文件
                            best_pt = run_dir / 'best.pt'
                            if best_pt.exists():
                                target_path = weights_dir / "dnf_yolo8m.pt"
                                shutil.copy(best_pt, target_path)
                                logger.info(f"已复制最佳模型到: {target_path}, 源文件: {best_pt}")
                                found = True
                                break
                            
                            # 查找所有权重文件
                            weights_files = list(run_dir.glob('*.pt'))
                            if weights_files:
                                # 按修改时间排序，获取最新的
                                latest_weights = max(weights_files, key=lambda x: x.stat().st_mtime)
                                target_path = weights_dir / "dnf_yolo8m.pt"
                                shutil.copy(latest_weights, target_path)
                                logger.info(f"已复制最新模型到: {target_path}, 源文件: {latest_weights}")
                                found = True
                                break
                    
                    if not found:
                        logger.warning(f"未能找到任何模型文件")
            except Exception as e:
                logger.warning(f"复制模型文件时出错: {e}")
                logger.warning("请手动复制最佳模型文件")
            
            logger.info("模型训练完成")
            return True
            
        except ImportError:
            logger.error("无法导入ultralytics，请安装: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"训练模型时出错: {e}")
            return False
    
    def export_model(self, format="onnx"):
        """
        导出模型为不同格式
        
        参数:
            format (str): 导出格式，可选 'onnx', 'torchscript', 'openvino', 等
        """
        try:
            from ultralytics import YOLO
            
            weights_dir = Path(self.config.get("weights_dir", "models/weights"))
            model_path = weights_dir / "dnf_yolo8m.pt"
            
            if not model_path.exists():
                logger.error(f"找不到训练好的模型: {model_path}")
                # 尝试查找其他可能的模型文件
                possible_models = list(weights_dir.glob('**/*.pt'))
                if possible_models:
                    model_path = possible_models[0]
                    logger.info(f"使用替代模型文件: {model_path}")
                else:
                    return False
            
            # 加载模型
            model = YOLO(model_path)
            
            # 导出模型
            logger.info(f"开始导出模型为 {format} 格式")
            
            export_path = model.export(format=format, imgsz=self.config.get("img_size", 640))
            
            logger.info(f"模型已导出为 {format} 格式: {export_path}")
            return True
            
        except ImportError:
            logger.error("无法导入ultralytics，请安装: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"导出模型时出错: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOv8模型训练工具")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="配置文件路径")
    parser.add_argument("--export", type=str, default=None, help="导出模型格式")
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YOLOv8Trainer(config_path=args.config)
    
    # 训练模型
    success = trainer.train_model()
    
    # 导出模型（如果指定）
    if success and args.export:
        trainer.export_model(format=args.export)

if __name__ == "__main__":
    main()