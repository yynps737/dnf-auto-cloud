#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练工具 - 用于训练YOLO模型识别DNF游戏中的对象
"""

import os
import sys
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
import shutil

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging_utils import setup_logging

# 设置日志
setup_logging(debug=True)
logger = logging.getLogger("ModelTrainer")

class YOLOTrainer:
    """YOLO模型训练器类"""
    
    def __init__(self, config_path="config/model_config.yaml"):
        """
        初始化训练器
        
        参数:
            config_path (str): 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # YOLO仓库路径
        self.yolo_repo = Path(self.config.get("yolo_repo", "yolov5"))
        
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
        
        # 创建YOLO数据配置
        data_yaml_path = data_dir / "data.yaml"
        
        # 统计图像数量
        images_dir = data_dir / "images"
        train_images = list((images_dir / "train").glob("*.jpg"))
        val_images = list((images_dir / "val").glob("*.jpg"))
        
        if not train_images:
            logger.warning(f"没有找到训练图像: {images_dir / 'train'}")
        
        if not val_images:
            logger.warning(f"没有找到验证图像: {images_dir / 'val'}")
        
        # 获取类别
        classes = self.config.get("classes", [
            "monster", "boss", "door", "item", "npc", "player", 
            "hp_bar", "mp_bar", "skill_ready", "cooldown"
        ])
        
        # 创建数据配置文件
        data_config = {
            "path": str(data_dir),
            "train": str(images_dir / "train"),
            "val": str(images_dir / "val"),
            "test": str(images_dir / "test"),
            "nc": len(classes),
            "names": classes
        }
        
        # 保存配置
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"已创建数据配置: {data_yaml_path}")
        logger.info(f"训练图像: {len(train_images)}, 验证图像: {len(val_images)}")
        
        return True
    
    def clone_yolo_repo(self):
        """克隆YOLOv5仓库"""
        if self.yolo_repo.exists():
            logger.info(f"YOLOv5仓库已存在: {self.yolo_repo}")
            return True
        
        try:
            logger.info(f"克隆YOLOv5仓库到: {self.yolo_repo}")
            subprocess.run(
                ["git", "clone", "https://github.com/ultralytics/yolov5", str(self.yolo_repo)],
                check=True
            )
            
            # 安装依赖
            subprocess.run(
                ["pip", "install", "-r", str(self.yolo_repo / "requirements.txt")],
                check=True
            )
            
            logger.info("YOLOv5仓库克隆完成并安装依赖")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"克隆YOLOv5仓库失败: {e}")
            return False
        except Exception as e:
            logger.error(f"准备YOLOv5仓库时出错: {e}")
            return False
    
    def train_model(self):
        """训练模型"""
        # 准备环境
        if not self.clone_yolo_repo():
            return False
        
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
        workers = self.config.get("workers", 8)
        
        # 选择预训练权重
        pretrained = self.config.get("pretrained", "yolov5m.pt")
        
        # 构建训练命令
        cmd = [
            "python", str(self.yolo_repo / "train.py"),
            "--img", str(img_size),
            "--batch", str(batch_size),
            "--epochs", str(epochs),
            "--data", str(data_yaml_path),
            "--weights", pretrained,
            "--workers", str(workers),
            "--project", str(weights_dir.parent),
            "--name", weights_dir.name,
            "--exist-ok"
        ]
        
        # 添加额外参数
        if self.config.get("cache", False):
            cmd.append("--cache")
        
        if self.config.get("device", ""):
            cmd.extend(["--device", self.config.get("device")])
        
        try:
            logger.info(f"开始训练模型，命令: {' '.join(cmd)}")
            
            # 运行训练
            subprocess.run(cmd, check=True)
            
            # 复制最终模型
            best_model = weights_dir / "weights" / "best.pt"
            if best_model.exists():
                target_path = weights_dir / "dnf_yolo8m.pt"
                shutil.copy(best_model, target_path)
                logger.info(f"已复制最佳模型到: {target_path}")
            
            logger.info("模型训练完成")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"训练模型失败: {e}")
            return False
        except Exception as e:
            logger.error(f"训练模型时出错: {e}")
            return False
    
    def export_model(self, format="onnx"):
        """
        导出模型为不同格式
        
        参数:
            format (str): 导出格式，可选 'onnx', 'torchscript', 'coreml'
        """
        weights_dir = Path(self.config.get("weights_dir", "models/weights"))
        model_path = weights_dir / "dnf_yolo8m.pt"
        
        if not model_path.exists():
            logger.error(f"找不到训练好的模型: {model_path}")
            return False
        
        # 构建导出命令
        cmd = [
            "python", str(self.yolo_repo / "export.py"),
            "--weights", str(model_path),
            "--include", format,
            "--img-size", str(self.config.get("img_size", 640)),
            "--batch-size", "1"
        ]
        
        try:
            logger.info(f"开始导出模型为 {format} 格式，命令: {' '.join(cmd)}")
            
            # 运行导出
            subprocess.run(cmd, check=True)
            
            logger.info(f"模型已导出为 {format} 格式")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"导出模型失败: {e}")
            return False
        except Exception as e:
            logger.error(f"导出模型时出错: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO模型训练工具")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="配置文件路径")
    parser.add_argument("--export", type=str, default=None, help="导出模型格式")
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YOLOTrainer(config_path=args.config)
    
    # 训练模型
    success = trainer.train_model()
    
    # 导出模型（如果指定）
    if success and args.export:
        trainer.export_model(format=args.export)

if __name__ == "__main__":
    main()