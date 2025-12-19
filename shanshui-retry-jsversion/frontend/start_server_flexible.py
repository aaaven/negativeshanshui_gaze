#!/usr/bin/env python3
"""
灵活的HTTP服务器，自动处理端口冲突
"""

import http.server
import socketserver
import os
import sys
import socket
from pathlib import Path

# 获取当前脚本所在目录
current_dir = Path(__file__).parent.absolute()

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(current_dir), **kwargs)
    
    def end_headers(self):
        # 添加CORS头，允许跨域请求
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def is_port_available(port):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port=8000, max_attempts=10):
    """查找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    return None

def start_server(port=None):
    """启动HTTP服务器"""
    if port is None:
        port = find_available_port()
        if port is None:
            print("错误：无法找到可用端口")
            return
    
    try:
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            print(f"前端服务器启动成功！")
            print(f"访问地址: http://localhost:{port}")
            print(f"服务目录: {current_dir}")
            print("按 Ctrl+C 停止服务器")
            print("-" * 50)
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n 服务器已停止")
                
    except OSError as e:
        if "Address already in use" in str(e) or "通常每个套接字地址" in str(e):
            print(f" 端口 {port} 已被占用")
            print(" 尝试使用其他端口...")
            start_server(port + 1)
        else:
            print(f" 启动服务器失败: {e}")

if __name__ == "__main__":
    # 检查命令行参数
    port = None
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("端口号必须是数字")
            sys.exit(1)
    
    start_server(port) 