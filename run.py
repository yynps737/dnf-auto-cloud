#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DNF自动化云服务主启动脚本
优化版 - 增加命令行参数和状态监控
"""

import os
import sys
import argparse
import logging
import time
import json
import platform
import subprocess
import threading
import requests
from datetime import datetime, timedelta

from server.main import start_server
from utils.logging_utils import setup_logging
from config.settings import SERVER, MODEL, LOGGING

# 版本信息
VERSION = "1.0.1"

def get_system_info():
    """获取系统信息"""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "architecture": platform.machine(),
        "hostname": platform.node()
    }
    
    # 检查CUDA和PyTorch
    try:
        import torch
        info["torch"] = torch.__version__
        
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        else:
            info["cuda"] = "Not available"
    except ImportError:
        info["torch"] = "Not installed"
        info["cuda"] = "Unknown"
    
    return info

def print_banner():
    """打印启动横幅"""
    banner = f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       DNF 自动化云服务 v{VERSION}                              ║
    ║                                                           ║
    ║       适用于地下城与勇士游戏的AI辅助系统                  ║
    ║       启动时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # 添加警告信息
    if MODEL.get("device", "") == "cpu":
        print("警告: 正在使用CPU模式运行，性能可能受限，建议使用GPU以获得更好的体验\n")

def monitor_server(port):
    """监控服务器状态"""
    try:
        # 定义状态监控线程
        def status_thread():
            start_time = time.time()
            
            while True:
                try:
                    current_time = time.time()
                    uptime = current_time - start_time
                    
                    # 读取server_info.json获取服务器信息
                    if os.path.exists("server_info.json"):
                        with open("server_info.json", "r") as f:
                            info = json.load(f)
                            
                        # 显示状态
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(f"\n==== DNF自动化云服务状态监控 ====")
                        print(f"服务器版本: {info.get('version', 'unknown')}")
                        print(f"启动时间: {info.get('started_at', 'unknown')}")
                        print(f"运行时长: {timedelta(seconds=int(uptime))}")
                        print(f"服务地址: {info.get('server', {}).get('url', 'unknown')}")
                        
                        # 尝试获取服务器状态
                        try:
                            status_file = "server_status.json"
                            if os.path.exists(status_file) and os.path.getmtime(status_file) > current_time - 60:
                                with open(status_file, "r") as sf:
                                    status = json.load(sf)
                                
                                print(f"\n==== 连接统计 ====")
                                print(f"活跃连接数: {status.get('connections', {}).get('active', 0)}")
                                print(f"总连接数: {status.get('connections', {}).get('total', 0)}")
                                
                                print(f"\n==== 性能统计 ====")
                                print(f"已处理图像: {status.get('performance', {}).get('images_processed', 0)}")
                                print(f"已生成动作: {status.get('performance', {}).get('actions_generated', 0)}")
                                print(f"错误数: {status.get('performance', {}).get('errors', 0)}")
                        except:
                            pass
                    
                    # 等待10秒更新一次
                    time.sleep(10)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"监控出错: {e}")
                    time.sleep(30)
        
        # 启动监控线程
        t = threading.Thread(target=status_thread, daemon=True)
        t.start()
        
    except Exception as e:
        print(f"启动监控失败: {e}")

def clean_old_logs():
    """清理旧的日志文件"""
    log_dir = os.path.dirname(LOGGING.get("file", ""))
    if not os.path.exists(log_dir):
        return
    
    try:
        # 获取所有日志文件
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        
        # 按修改时间排序
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
        
        # 如果超过10个日志文件，删除最旧的
        if len(log_files) > 10:
            for f in log_files[:-10]:
                try:
                    os.remove(os.path.join(log_dir, f))
                    print(f"已删除旧日志文件: {f}")
                except:
                    pass
    except Exception as e:
        print(f"清理日志时出错: {e}")

def check_updates():
    """检查更新"""
    try:
        # 这里可以实现检查更新的逻辑
        # 例如，检查Git仓库或向API服务器查询
        pass
    except:
        pass

def create_status_api(port):
    """创建一个简单的状态API服务器"""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading
        import json
        
        class StatusHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    
                    # 读取服务器状态
                    status = {"status": "running"}
                    if os.path.exists("server_status.json"):
                        try:
                            with open("server_status.json", "r") as f:
                                status = json.load(f)
                        except:
                            pass
                    
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        def run_server():
            server_address = ("", port + 1)  # 使用主端口+1
            httpd = HTTPServer(server_address, StatusHandler)
            print(f"状态API服务器已启动: http://localhost:{port + 1}/status")
            httpd.serve_forever()
        
        # 启动API服务器线程
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        
    except Exception as e:
        print(f"启动状态API服务器失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"DNF自动化云服务 v{VERSION}")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU模式")
    parser.add_argument("--no-monitor", action="store_true", help="禁用状态监控")
    parser.add_argument("--version", action="version", version=f"DNF自动化云服务 v{VERSION}")
    args = parser.parse_args()
    
    # 开启状态监控
    if not args.no_monitor:
        monitor_server(args.port)
        create_status_api(args.port)
    
    # 打印横幅
    print_banner()
    
    # 清理旧日志
    clean_old_logs()
    
    # 检查更新
    check_updates()
    
    # 强制CPU模式
    if args.cpu:
        MODEL["device"] = "cpu"
        print("已强制使用CPU模式")
    
    # 设置日志
    setup_logging(debug=args.debug)
    logger = logging.getLogger("DNFAutoCloud")
    
    # 打印系统信息
    system_info = get_system_info()
    logger.info(f"系统信息: {json.dumps(system_info, ensure_ascii=False)}")
    
    # 启动服务
    logger.info(f"正在启动DNF自动化云服务 v{VERSION}...")
    
    try:
        # 启动服务器
        success = start_server(host=args.host, port=args.port, debug=args.debug)
        
        if not success:
            logger.error("服务器启动失败")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("用户中断，服务已停止")
    except Exception as e:
        logger.critical(f"启动服务时出错: {e}", exc_info=True)
        sys.exit(1)