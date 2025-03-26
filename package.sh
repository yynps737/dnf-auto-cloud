#!/bin/bash
# 创建客户端打包目录
mkdir -p dnf-client-package
cp client.py dnf-client-package/
cp -r win_libs/* dnf-client-package/

# 创建默认配置文件
cat > dnf-client-package/config.ini << EOF
[Server]
url = wss://your-server-ip:8080/ws
verify_ssl = false

[Capture]
interval = 0.5
quality = 70

[Game]
window_title = 地下城与勇士
key_mapping = default
movement_smoothness = 0.8

[Connection]
max_retries = 5
retry_delay = 5
heartbeat_interval = 5
EOF

# 创建启动脚本
cat > dnf-client-package/start.bat << EOF
@echo off
echo 正在启动DNF自动化客户端...
python client.py
pause
EOF

# 打包为zip
cd dnf-client-package
zip -r ../dnf-client-windows.zip *
cd ..

echo "打包完成: dnf-client-windows.zip"