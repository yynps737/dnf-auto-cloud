#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安全和加密工具，提供加密通信和认证功能
"""

import os
import json
import base64
import logging
import random
import string
import ssl
import subprocess
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

from config.settings import SECURITY, BASE_DIR

logger = logging.getLogger("DNFAutoCloud")

# 生成或加载加密密钥
def get_encryption_key():
    """获取或生成加密密钥"""
    key_file = os.path.join(BASE_DIR, "config", "encryption.key")
    
    if os.path.exists(key_file):
        with open(key_file, "rb") as f:
            key = f.read()
    else:
        # 生成新密钥
        key = Fernet.generate_key()
        # 保存密钥
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
        with open(key_file, "wb") as f:
            f.write(key)
        
        logger.info(f"已生成新的加密密钥: {key_file}")
    
    return key

# 初始化加密器
_encryption_key = get_encryption_key()
_cipher = Fernet(_encryption_key)

def encrypt_message(message):
    """
    加密消息
    
    参数:
        message (dict/str): 要加密的消息
        
    返回:
        str: 加密后的Base64字符串
    """
    try:
        if isinstance(message, dict):
            message = json.dumps(message)
        
        # 转换为字节
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # 加密
        encrypted = _cipher.encrypt(message)
        
        # 转换为Base64字符串
        return base64.b64encode(encrypted).decode('utf-8')
        
    except Exception as e:
        logger.error(f"加密消息时出错: {e}")
        # 无法加密时返回原始JSON
        if isinstance(message, dict):
            return json.dumps(message)
        return message

def decrypt_message(encrypted_message):
    """
    解密消息
    
    参数:
        encrypted_message (str): 加密的Base64字符串
        
    返回:
        dict: 解密后的消息
    """
    try:
        # 检查是否已经是JSON（未加密）
        try:
            return json.loads(encrypted_message)
        except json.JSONDecodeError:
            pass
        
        # 解码Base64
        encrypted_bytes = base64.b64decode(encrypted_message)
        
        # 解密
        decrypted = _cipher.decrypt(encrypted_bytes)
        
        # 解析JSON
        return json.loads(decrypted.decode('utf-8'))
        
    except Exception as e:
        logger.error(f"解密消息时出错: {e}")
        # 尝试直接解析JSON
        try:
            return json.loads(encrypted_message)
        except:
            # 无法解密或解析时返回空字典
            return {}

def generate_ssl_cert():
    """生成自签名SSL证书"""
    cert_file = SECURITY["ssl_cert"]
    key_file = SECURITY["ssl_key"]
    
    # 确保目录存在
    os.makedirs(os.path.dirname(cert_file), exist_ok=True)
    
    # 生成私钥
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    # 保存私钥
    with open(key_file, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # 使用OpenSSL命令生成证书
    cmd = [
        "openssl", "req", "-new", "-x509", "-key", key_file,
        "-out", cert_file, "-days", "365",
        "-subj", "/CN=localhost"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"已生成自签名SSL证书: {cert_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"生成SSL证书时出错: {e}")
        raise