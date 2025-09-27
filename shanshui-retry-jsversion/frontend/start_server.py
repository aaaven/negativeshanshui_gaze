#!/usr/bin/env python3
"""
简单的HTTP服务器，用于提供前端文件
"""

import http.server
import socketserver
import os
import sys
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

def start_server(port=8000):
    """启动HTTP服务器"""
    with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
        print(f"前端服务器启动在 http://localhost:{port}")
        print(f"请在浏览器中打开 http://localhost:{port}")
        print("按 Ctrl+C 停止服务器")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("端口号必须是数字")
            sys.exit(1)
    
    start_server(port)