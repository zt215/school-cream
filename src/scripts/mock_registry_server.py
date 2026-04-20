#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的测试服务器，用于提供课程数据 API。"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# 测试数据
TEST_DATA = [
    {
        "timeAdd": "14:30:00",
        "courseName": "高等数学",
        "room_id": "A101",
        "device": [
            {
                "name": "前摄像头",
                "type": "camera",
                "liveUrl": "0",  # 使用0表示默认摄像头，或者提供视频文件路径
                "model_number": "IPC-123",
                "host": "192.168.1.100",
                "serial_number": "SN123456789",
                "token": "auth_token_here",
                "admin": "admin",
                "password": "password123",
                "create_by": "system",
                "create_date": "2024-01-01 00:00:00",
                "update_by": "admin",
                "update_date": "2024-01-01 00:00:00",
                "status": "1",
                "remarks": "前门摄像头",
                "del_flag": "0",
            }
        ],
    }
]


class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/courses" or self.path == "/courses":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(TEST_DATA, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, fmt, *args):
        # 简化日志输出
        print(f"[{self.address_string()}] {args[0]}")


def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, APIHandler)
    print(f"测试服务器运行在 http://localhost:{port}/api/courses")
    print("按 Ctrl+C 停止服务器")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        httpd.shutdown()


def main() -> None:
    run_server()


if __name__ == "__main__":
    main()

