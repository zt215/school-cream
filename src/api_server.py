#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
青衿智眼 API 服务
- 提供实时状态、历史、告警、统计接口
- 供前端页面和外部插件调用
"""

from collections import deque
from datetime import datetime
import os
import threading
import time
from pathlib import Path
import random
import sqlite3
from functools import wraps
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response, redirect, url_for, session, send_file
from flask_cors import CORS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from app.api.routes.auth_health import (
    handle_health,
    handle_login,
    handle_logout,
    handle_register,
)
from app.api.routes.status import (
    handle_focus_realtime,
    handle_get_alerts,
    handle_get_history,
    handle_get_stats,
    handle_get_status,
)
from app.api.routes.media import handle_get_latest_frame, handle_stream_frames
from app.api.routes.system import handle_start_system, handle_stop_system
from app.api.routes.assets import handle_export_pdf, handle_list_models, handle_upload_file

SYSTEM_NAME = os.getenv("SYSTEM_NAME", "大学生课堂行为识别系统")

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "school-cream-secret-key")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
AUTH_DB = Path(os.getenv("AUTH_DB_PATH", "data/auth.db"))
AUTH_DB.parent.mkdir(parents=True, exist_ok=True)


def init_auth_db():
    conn = sqlite3.connect(str(AUTH_DB))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def require_login(api=False):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if session.get("user_id"):
                return fn(*args, **kwargs)
            if api:
                return jsonify({"success": False, "error": "请先登录"}), 401
            return redirect(url_for("login_page"))
        return wrapper
    return deco


def compute_focus_score(counts):
    """基于行为统计计算课堂专注度（0-100）。"""
    person = max(0, int(counts.get("Person", 0)))
    raise_hand = max(0, int(counts.get("Raise Hand", 0)))
    lie_down = max(0, int(counts.get("Lie Down", 0)))
    phone_usage = max(0, int(counts.get("Phone Usage", 0)))

    # 规则融合：举手是正向信号，躺下/手机是负向信号。
    base = 80.0
    if person > 0:
        active_ratio = min(1.0, raise_hand / person)
        lie_ratio = min(1.0, lie_down / person)
        phone_ratio = min(1.0, phone_usage / person)
        score = base + 20.0 * active_ratio - 30.0 * lie_ratio - 35.0 * phone_ratio
    else:
        score = base

    score = max(0.0, min(100.0, score))
    if score >= 80:
        level = "高"
    elif score >= 60:
        level = "中"
    else:
        level = "低"
    return round(score, 2), level


detection_data = {
    "current": {"Person": 0, "Raise Hand": 0, "Lie Down": 0, "Phone Usage": 0},
    "history": deque(maxlen=1000),
    "alerts": deque(maxlen=300),
    "camera_info": {"name": "未知摄像头", "course_name": "未知课程", "room_id": "未知教室", "status": "stopped"},
    "last_update": None,
    "system_status": {"running": False, "frame_count": 0, "fps": 0},
    "focus": {"score": 80.0, "level": "中", "source": "rule-based-v1"},
    # 多模态占位：视觉已接入，音频/交互可后续补充。
    "multimodal": {"vision": True, "audio": False, "interaction": False},
    "demo_mode": True,
}
data_lock = threading.Lock()

active_processor = None
video_thread = None
processor_lock = threading.Lock()
video_thread_lock = threading.Lock()
last_preview_at = 0.0
PREVIEW_TTL_SEC = 120.0


def _set_video_thread(value):
    global video_thread
    video_thread = value


def _set_active_processor(value):
    global active_processor
    active_processor = value


def _set_last_preview_at(value):
    global last_preview_at
    last_preview_at = value


def update_detection_data(counts, camera_info=None):
    with data_lock:
        now_iso = datetime.now().isoformat()
        detection_data["current"] = counts.copy()
        detection_data["last_update"] = now_iso
        if camera_info:
            detection_data["camera_info"].update(camera_info)

        score, level = compute_focus_score(counts)
        detection_data["focus"] = {
            "score": score,
            "level": level,
            "source": "rule-based-v1",
            "updated_at": now_iso,
        }

        detection_data["history"].append(
            {
                "timestamp": now_iso,
                "counts": counts.copy(),
                "camera_info": detection_data["camera_info"].copy(),
                "focus": detection_data["focus"].copy(),
            }
        )

        if counts.get("Lie Down", 0) > 0 or counts.get("Phone Usage", 0) > 0:
            detection_data["alerts"].append(
                {
                    "timestamp": now_iso,
                    "type": "异常行为",
                    "details": {
                        "Lie Down": counts.get("Lie Down", 0),
                        "Phone Usage": counts.get("Phone Usage", 0),
                    },
                    "focus_score": score,
                    "focus_level": level,
                    "is_resolved": False,
                }
            )


def update_system_status(status_info):
    with data_lock:
        detection_data["system_status"].update(status_info)


def seed_demo_data(points: int = 12):
    """
    预填充一批演示数据，便于前端在未启动监控时展示 UI。
    一旦开始真实监控，VideoProcessor 会持续更新这些数据。
    """
    with data_lock:
        # 避免覆盖已有真实数据
        if detection_data["history"]:
            return
        camera_info = {
            "name": "示例摄像头-前门",
            "course_name": "示例课程",
            "room_id": "A101",
            "status": "stopped",
        }
        detection_data["camera_info"] = camera_info.copy()
        detection_data["system_status"] = {"running": False, "frame_count": 0, "fps": 0}

    # 注意：这里不能持有 data_lock 再调用 update_detection_data（会死锁）
    for _ in range(points):
        # 演示数据：总人数固定为 48，方便前端展示
        person = 48
        raise_hand = random.randint(0, 4)
        lie_down = random.randint(0, 2)
        phone = random.randint(0, 3)
        counts = {"Person": person, "Raise Hand": raise_hand, "Lie Down": lie_down, "Phone Usage": phone}
        update_detection_data(counts, camera_info)

    with data_lock:
        detection_data["last_update"] = datetime.now().isoformat()


@app.route("/", methods=["GET"])
@require_login(api=False)
def index_page():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login_page():
    return handle_login(AUTH_DB)


@app.route("/register", methods=["GET", "POST"])
def register_page():
    return handle_register(AUTH_DB)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    return handle_logout()


@app.route("/api/status", methods=["GET"])
@require_login(api=True)
def get_status():
    return handle_get_status(detection_data, data_lock, seed_demo_data)


@app.route("/api/history", methods=["GET"])
@require_login(api=True)
def get_history():
    return handle_get_history(detection_data, data_lock)


@app.route("/api/alerts", methods=["GET"])
@require_login(api=True)
def get_alerts():
    return handle_get_alerts(detection_data, data_lock)


@app.route("/api/stats", methods=["GET"])
@require_login(api=True)
def get_stats():
    return handle_get_stats(detection_data, data_lock)


@app.route("/api/focus/realtime", methods=["GET"])
@require_login(api=True)
def focus_realtime():
    return handle_focus_realtime(detection_data, data_lock)


@app.route("/api/frame", methods=["GET"])
@require_login(api=True)
def get_latest_frame():
    return handle_get_latest_frame(
        processor_lock=processor_lock,
        get_active_processor=lambda: active_processor,
        data_lock=data_lock,
        detection_data=detection_data,
        get_last_preview_at=lambda: last_preview_at,
        preview_ttl_sec=PREVIEW_TTL_SEC,
    )


@app.route("/api/stream", methods=["GET"])
@require_login(api=True)
def stream_frames():
    return handle_stream_frames(processor_lock=processor_lock, get_active_processor=lambda: active_processor)


@app.route("/api/system/start", methods=["POST"])
@require_login(api=True)
def start_system():
    global video_thread, active_processor
    return handle_start_system(
        video_thread_lock=video_thread_lock,
        processor_lock=processor_lock,
        data_lock=data_lock,
        detection_data=detection_data,
        get_video_thread=lambda: video_thread,
        set_video_thread=lambda value: _set_video_thread(value),
        get_active_processor=lambda: active_processor,
        set_active_processor=lambda value: _set_active_processor(value),
        set_last_preview_at=lambda value: _set_last_preview_at(value),
        update_system_status=update_system_status,
    )


@app.route("/api/system/stop", methods=["POST"])
@require_login(api=True)
def stop_system():
    global video_thread, active_processor
    return handle_stop_system(
        video_thread_lock=video_thread_lock,
        processor_lock=processor_lock,
        data_lock=data_lock,
        detection_data=detection_data,
        get_video_thread=lambda: video_thread,
        set_video_thread=lambda value: _set_video_thread(value),
        get_active_processor=lambda: active_processor,
        update_system_status=update_system_status,
    )


@app.route("/api/upload", methods=["POST"])
@require_login(api=True)
def upload_file():
    return handle_upload_file(UPLOAD_DIR)


@app.route("/api/models", methods=["GET"])
@require_login(api=True)
def list_models():
    return handle_list_models(MODEL_DIR)


@app.route("/api/export/pdf", methods=["GET"])
@require_login(api=True)
def export_pdf():
    return handle_export_pdf(detection_data, data_lock)


@app.route("/health", methods=["GET"])
def health():
    return handle_health(SYSTEM_NAME)


def run_api_server(host="0.0.0.0", port=5000, debug=False):
    # 开启 threaded，避免 /api/stream 长连接阻塞其它接口
    init_auth_db()
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    run_api_server()
