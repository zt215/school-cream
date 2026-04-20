#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
青衿智眼 API 服务
- 提供实时状态、历史、告警、统计接口
- 供前端页面和外部插件调用
"""

from collections import deque
from datetime import datetime, timedelta
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
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# 与推理端共享配置（main.py 通过 `from config import DETECTION_CONFIG` 引用同一个 dict）
import config as app_config

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

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


DEFAULT_STUDENT_ROSTER = [
    {"student_id": "02329001", "name": "董文武"},
    {"student_id": "02329002", "name": "皇宫杰"},
    {"student_id": "02329003", "name": "贾咏霖"},
    {"student_id": "02329004", "name": "何晓东"},
    {"student_id": "02329005", "name": "刘泽宇"},
    {"student_id": "02329006", "name": "周炳甲"},
    {"student_id": "02329007", "name": "范钰乐"},
    {"student_id": "02329008", "name": "宫歆"},
    {"student_id": "02329010", "name": "刘硕"},
    {"student_id": "02329011", "name": "张浩然"},
    {"student_id": "02329012", "name": "王浩然"},
    {"student_id": "02329019", "name": "穆晓鹏"},
    {"student_id": "02329020", "name": "刘漫"},
    {"student_id": "02329022", "name": "张跃"},
    {"student_id": "02329023", "name": "王子帅"},
    {"student_id": "02329025", "name": "韩尚儒"},
    {"student_id": "02329028", "name": "郭靖"},
    {"student_id": "02329029", "name": "刘天宇"},
    {"student_id": "02329030", "name": "郭彦廷"},
    {"student_id": "02329032", "name": "卢嘉峻"},
    {"student_id": "02329034", "name": "万钰雨"},
    {"student_id": "02329036", "name": "孙文博"},
    {"student_id": "02329039", "name": "边志源"},
    {"student_id": "02329040", "name": "杨云凯"},
    {"student_id": "02329042", "name": "赵德昌"},
    {"student_id": "02329043", "name": "侯洪贤"},
    {"student_id": "02329044", "name": "赵鑫"},
    {"student_id": "02329047", "name": "杨志伟"},
    {"student_id": "02329048", "name": "杜嘉泽"},
    {"student_id": "02329049", "name": "屈宝国"},
    {"student_id": "02329050", "name": "谢佳乐"},
    {"student_id": "02329054", "name": "周涛"},
    {"student_id": "02329057", "name": "宫翊"},
    {"student_id": "02329058", "name": "减志坚"},
    {"student_id": "02329059", "name": "郑世玛"},
    {"student_id": "02329009", "name": "王思慧"},
    {"student_id": "02329013", "name": "李金"},
    {"student_id": "02329014", "name": "金佳莹"},
    {"student_id": "02329015", "name": "陈文娇"},
    {"student_id": "02329017", "name": "杨安琪"},
    {"student_id": "02329018", "name": "张懿涓"},
    {"student_id": "02329021", "name": "王鑫"},
    {"student_id": "02329024", "name": "薛倩"},
    {"student_id": "02329026", "name": "张嘉怡"},
    {"student_id": "02329027", "name": "赵嘉琪"},
    {"student_id": "02329031", "name": "司佳音"},
    {"student_id": "02329033", "name": "秦悦"},
    {"student_id": "02329037", "name": "高熙"},
    {"student_id": "02329038", "name": "姜阳"},
    {"student_id": "02329041", "name": "王仕欣"},
    {"student_id": "02329045", "name": "郭婉慈"},
    {"student_id": "02329046", "name": "路倩"},
    {"student_id": "02329051", "name": "王煜"},
    {"student_id": "02329052", "name": "贾娜娜"},
    {"student_id": "02329055", "name": "赵金梅"},
    {"student_id": "02329056", "name": "杨朝敏"},
    {"student_id": "02329064", "name": "杜梓怡"},
    {"student_id": "02329014", "name": "焦倩倩"},
]


def build_attendance_state(roster):
    clean_roster = []
    seen = set()
    for item in roster:
        sid = str(item.get("student_id", "")).strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        clean_roster.append({"student_id": sid, "name": item.get("name") or sid})
    return {
        "roster": clean_roster,
        "records": [],
        "present_ids": [],
        "absent_ids": [x["student_id"] for x in clean_roster],
        "late_ids": [],
        "summary": {
            "expected": len(clean_roster),
            "present": 0,
            "absent": len(clean_roster),
            "late": 0,
            "rate": 0.0,
            "status": "未开始",
        },
    }


def refresh_attendance(counts):
    attendance = detection_data.get("attendance")
    if not attendance:
        return
    roster = attendance.get("roster", [])
    expected = len(roster)
    present_n = min(max(0, int(counts.get("Person", 0))), expected)
    present_students = roster[:present_n]
    absent_students = roster[present_n:]
    present_ids = [x["student_id"] for x in present_students]
    absent_ids = [x["student_id"] for x in absent_students]

    attendance["present_ids"] = present_ids
    attendance["absent_ids"] = absent_ids
    attendance["records"] = [
        {"student_id": x["student_id"], "name": x["name"], "status": "present"} for x in present_students
    ] + [
        {"student_id": x["student_id"], "name": x["name"], "status": "absent"} for x in absent_students
    ]
    rate = (present_n / expected * 100.0) if expected > 0 else 0.0
    attendance["summary"] = {
        "expected": expected,
        "present": present_n,
        "absent": max(0, expected - present_n),
        "late": 0,
        "rate": round(rate, 1),
        "status": "签到中" if detection_data.get("system_status", {}).get("running") else "未开始",
    }


def beijing_now():
    """返回北京时间，确保课程匹配按东八区进行。"""
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Asia/Shanghai"))
    # 兼容极少数不支持 zoneinfo 的运行环境
    return datetime.utcnow() + timedelta(hours=8)


def load_students_from_db(conn):
    cur = conn.execute(
        """
        SELECT student_id, name, COALESCE(gender, ''), COALESCE(class_name, ''), status
        FROM students
        ORDER BY student_id
        """
    )
    rows = cur.fetchall()
    return [
        {
            "student_id": str(r[0]),
            "name": r[1] or str(r[0]),
            "gender": r[2] or "",
            "class_name": r[3] or "",
            "status": r[4] or "active",
        }
        for r in rows
    ]


def active_roster(students):
    return [x for x in students if (x.get("status") or "active") == "active"]


def seed_students_if_empty(conn):
    conn.executemany(
        """
        INSERT OR IGNORE INTO students (student_id, name, gender, class_name, status, created_at)
        VALUES (?, ?, ?, ?, 'active', ?)
        """,
        [
            (
                x["student_id"],
                x["name"],
                x.get("gender", ""),
                x.get("class_name", "2023级 软件工程1班"),
                datetime.now().isoformat(),
            )
            for x in DEFAULT_STUDENT_ROSTER
        ],
    )


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                gender TEXT,
                class_name TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            )
            """
        )
        seed_students_if_empty(conn)
        conn.commit()

        students = load_students_from_db(conn)
        roster = active_roster(students)
        if not roster:
            roster = DEFAULT_STUDENT_ROSTER
        attendance_state = build_attendance_state(roster)
        with data_lock:
            detection_data["attendance"] = attendance_state
            detection_data["students"] = students
            detection_data["camera_info"]["class_size"] = attendance_state["summary"]["expected"]
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
    "camera_info": {
        "name": "未知摄像头",
        "course_name": "未知课程",
        "class_name": "未知班级",
        "teacher_name": "未知老师",
        "class_time": "未知时间",
        "room_id": "未知教室",
        "status": "stopped"
    },
    "last_update": None,
    "system_status": {"running": False, "frame_count": 0, "fps": 0},
    "focus": {"score": 80.0, "level": "中", "source": "rule-based-v1"},
    # 多模态占位：视觉已接入，音频/交互可后续补充。
    "multimodal": {"vision": True, "audio": False, "interaction": False},
    "attendance": build_attendance_state([]),
    "students": [],
    "demo_mode": True,
}
data_lock = threading.Lock()

active_processor = None
video_thread = None
processor_lock = threading.Lock()
video_thread_lock = threading.Lock()
last_preview_at = 0.0
PREVIEW_TTL_SEC = 120.0


def update_detection_data(counts, camera_info=None):
    with data_lock:
        # 一旦有真实推理数据写入，关闭演示模式（否则 /api/status 会持续 seed_demo_data 覆盖展示）
        detection_data["demo_mode"] = False
        now_iso = datetime.now().isoformat()
        detection_data["current"] = counts.copy()
        detection_data["last_update"] = now_iso
        if camera_info:
            detection_data["camera_info"].update(camera_info)
        refresh_attendance(counts)

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
        if detection_data.get("attendance"):
            detection_data["attendance"]["summary"]["status"] = "签到中" if detection_data["system_status"].get("running") else "未开始"


def probe_camera_devices(max_index=20):
    """
    探测本机可用摄像头索引，返回可读帧的设备列表。
    Windows 下优先使用 CAP_DSHOW，减少黑屏/占用兼容问题。
    """
    max_index = max(0, min(int(max_index), 30))
    devices = []

    for idx in range(max_index + 1):
        backends = [("default", None)]
        if os.name == "nt":
            backends = [("dshow", cv2.CAP_DSHOW), ("default", None)]

        picked = None
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue

            ok = False
            frame = None
            # 预热几帧，避免首帧为空
            for _ in range(4):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.04)

            if ok and frame is not None:
                height, width = frame.shape[:2]
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                picked = {
                    "index": idx,
                    "backend": backend_name,
                    "width": int(width),
                    "height": int(height),
                    "fps": round(fps, 2),
                }
                cap.release()
                break

            cap.release()

        if picked:
            devices.append(picked)

    return devices


def seed_demo_data(points: int = 12):
    """
    预填充一批演示数据，便于前端在未启动监控时展示 UI。
    一旦开始真实监控，VideoProcessor 会持续更新这些数据。
    """
    with data_lock:
        # 避免覆盖已有真实数据
        if detection_data["history"]:
            return
        sample_timetable = [
            {"course_name": "软件质量保证与测试", "class_name": "2023级 软件工程1班", "teacher_name": "冯欣", "class_time": "周一 08:00-09:50（1-2节）", "room_id": "C-219", "weekday": 0, "start_min": 8 * 60, "end_min": 9 * 60 + 50},
            {"course_name": "计算机机房建设与维护", "class_name": "2023级 软件工程1班", "teacher_name": "杨书峰", "class_time": "周二 10:10-12:00（3-4节）", "room_id": "A-201", "weekday": 1, "start_min": 10 * 60 + 10, "end_min": 12 * 60},
            {"course_name": "软件工程综合实践", "class_name": "2023级 软件工程1班", "teacher_name": "戴海滨", "class_time": "周三 08:00-09:50（1-2节）", "room_id": "A-503", "weekday": 2, "start_min": 8 * 60, "end_min": 9 * 60 + 50},
            {"course_name": "就业指导与职业发展", "class_name": "2023级 软件工程1班", "teacher_name": "侯辰", "class_time": "周三 14:30-16:20（5-6节）", "room_id": "A-509", "weekday": 2, "start_min": 14 * 60 + 30, "end_min": 16 * 60 + 20},
            {"course_name": "Python程序设计语言", "class_name": "2023级 软件工程1班", "teacher_name": "张宏琳", "class_time": "周四 14:30-17:30（5-8节）", "room_id": "A-205", "weekday": 3, "start_min": 14 * 60 + 30, "end_min": 17 * 60 + 30},
        ]
        now = beijing_now()
        current_weekday = now.weekday()
        current_min = now.hour * 60 + now.minute

        # 优先匹配当前时间所在课程，其次匹配当天最近课程，最后随机一个
        current_course = next(
            (x for x in sample_timetable if x["weekday"] == current_weekday and x["start_min"] <= current_min <= x["end_min"]),
            None,
        )
        if current_course is None:
            today_courses = [x for x in sample_timetable if x["weekday"] == current_weekday]
            if today_courses:
                current_course = min(today_courses, key=lambda x: abs(x["start_min"] - current_min))
            else:
                current_course = random.choice(sample_timetable)
        picked = current_course
        camera_info = {
            "name": "示例摄像头-前门",
            "course_name": picked["course_name"],
            "class_name": picked["class_name"],
            "teacher_name": picked["teacher_name"],
            "class_time": picked["class_time"],
            "room_id": picked["room_id"],
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
    if request.method == "GET":
        if session.get("user_id"):
            return redirect(url_for("index_page"))
        return render_template("login.html", error="")

    payload = request.form or (request.get_json(silent=True) or {})
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    if not username or not password:
        return render_template("login.html", error="用户名和密码不能为空"), 400

    conn = sqlite3.connect(str(AUTH_DB))
    try:
        cur = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
    finally:
        conn.close()
    if not row or not check_password_hash(row[2], password):
        return render_template("login.html", error="用户名或密码错误"), 401

    session["user_id"] = row[0]
    session["username"] = row[1]
    return redirect(url_for("index_page"))


@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "GET":
        return render_template("register.html", error="")

    payload = request.form or (request.get_json(silent=True) or {})
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    confirm_password = (payload.get("confirm_password") or "").strip()
    if len(username) < 3 or len(password) < 6:
        return render_template("register.html", error="用户名至少3位，密码至少6位"), 400
    if password != confirm_password:
        return render_template("register.html", error="两次输入密码不一致"), 400

    conn = sqlite3.connect(str(AUTH_DB))
    try:
        cur = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return render_template("register.html", error="用户名已存在"), 400
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for("login_page"))


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/api/status", methods=["GET"])
@require_login(api=True)
def get_status():
    if detection_data.get("demo_mode"):
        seed_demo_data()
    with data_lock:
        # deque 不能直接 JSON 序列化，这里转成 list 返回
        data = {
            **detection_data,
            "history": list(detection_data["history"]),
            "alerts": list(detection_data["alerts"]),
        }
        return jsonify({"success": True, "data": data})


@app.route("/api/attendance", methods=["GET"])
@require_login(api=True)
def attendance_status():
    limit = request.args.get("limit", 30, type=int)
    with data_lock:
        attendance = detection_data.get("attendance", {})
        records = attendance.get("records", [])
        records = records[: max(0, limit)]
        payload = {
            "summary": attendance.get("summary", {}),
            "records": records,
            "present_ids": attendance.get("present_ids", []),
            "absent_ids": attendance.get("absent_ids", []),
            "late_ids": attendance.get("late_ids", []),
        }
    return jsonify({"success": True, "data": payload})


@app.route("/api/students", methods=["GET"])
@require_login(api=True)
def get_students():
    status = (request.args.get("status", "") or "").strip()
    conn = sqlite3.connect(str(AUTH_DB))
    try:
        students = load_students_from_db(conn)
    finally:
        conn.close()
    if status:
        students = [x for x in students if (x.get("status") or "active") == status]
    return jsonify({"success": True, "data": students, "total": len(students)})


@app.route("/api/students/import", methods=["POST"])
@require_login(api=True)
def import_students():
    payload = request.get_json(silent=True) or {}
    students = payload.get("students")
    replace = bool(payload.get("replace", False))
    if not isinstance(students, list) or not students:
        return jsonify({"success": False, "error": "students 不能为空，且必须为数组"}), 400

    now_iso = datetime.now().isoformat()
    normalized = []
    seen = set()
    for item in students:
        sid = str((item or {}).get("student_id", "")).strip()
        name = str((item or {}).get("name", "")).strip()
        if not sid or not name or sid in seen:
            continue
        seen.add(sid)
        normalized.append(
            {
                "student_id": sid,
                "name": name,
                "gender": str((item or {}).get("gender", "")).strip(),
                "class_name": str((item or {}).get("class_name", "")).strip(),
                "status": str((item or {}).get("status", "active")).strip() or "active",
            }
        )

    if not normalized:
        return jsonify({"success": False, "error": "没有可导入的数据（请检查学号/姓名）"}), 400

    conn = sqlite3.connect(str(AUTH_DB))
    try:
        if replace:
            conn.execute("DELETE FROM students")
        conn.executemany(
            """
            INSERT INTO students (student_id, name, gender, class_name, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                name = excluded.name,
                gender = excluded.gender,
                class_name = excluded.class_name,
                status = excluded.status
            """,
            [
                (x["student_id"], x["name"], x["gender"], x["class_name"], x["status"], now_iso)
                for x in normalized
            ],
        )
        conn.commit()

        all_students = load_students_from_db(conn)
        roster = active_roster(all_students)
    finally:
        conn.close()

    attendance_state = build_attendance_state(roster)
    with data_lock:
        detection_data["students"] = all_students
        detection_data["attendance"] = attendance_state
        detection_data["camera_info"]["class_size"] = attendance_state["summary"]["expected"]

    return jsonify(
        {
            "success": True,
            "data": {
                "imported": len(normalized),
                "students_total": len(all_students),
                "active_total": len(roster),
            },
        }
    )


@app.route("/api/history", methods=["GET"])
@require_login(api=True)
def get_history():
    limit = request.args.get("limit", 100, type=int)
    with data_lock:
        items = list(detection_data["history"])[-limit:]
    return jsonify({"success": True, "data": items, "total": len(items)})


@app.route("/api/alerts", methods=["GET"])
@require_login(api=True)
def get_alerts():
    limit = request.args.get("limit", 50, type=int)
    with data_lock:
        items = list(detection_data["alerts"])[-limit:]
    return jsonify({"success": True, "data": items, "total": len(items)})


@app.route("/api/stats", methods=["GET"])
@require_login(api=True)
def get_stats():
    with data_lock:
        history = list(detection_data["history"])
        if not history:
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "total_frames": 0,
                        "total_person": 0,
                        "total_raise_hand": 0,
                        "total_lie_down": 0,
                        "total_phone_usage": 0,
                        "focus_avg": detection_data["focus"]["score"],
                    },
                }
            )
        total_person = sum(r["counts"].get("Person", 0) for r in history)
        total_raise = sum(r["counts"].get("Raise Hand", 0) for r in history)
        total_lie = sum(r["counts"].get("Lie Down", 0) for r in history)
        total_phone = sum(r["counts"].get("Phone Usage", 0) for r in history)
        focus_avg = round(sum(r.get("focus", {}).get("score", 80) for r in history) / len(history), 2)
    return jsonify(
        {
            "success": True,
            "data": {
                "total_frames": len(history),
                "total_person": total_person,
                "total_raise_hand": total_raise,
                "total_lie_down": total_lie,
                "total_phone_usage": total_phone,
                "focus_avg": focus_avg,
            },
        }
    )


@app.route("/api/focus/realtime", methods=["GET"])
@require_login(api=True)
def focus_realtime():
    with data_lock:
        focus = detection_data["focus"]
        modal = detection_data["multimodal"]
    return jsonify({"success": True, "data": {"focus": focus, "multimodal": modal}})


@app.route("/api/frame", methods=["GET"])
@require_login(api=True)
def get_latest_frame():
    """
    返回最新标注帧（JPEG）。
    前端轮询该接口实现“监控预览”。
    """
    with processor_lock:
        proc = active_processor
    if not proc:
        return Response(status=204)
    # 仅在非运行态（图片单帧预览）应用 TTL；运行态视频流不限制
    with data_lock:
        running = bool(detection_data.get("system_status", {}).get("running", False))
    if (not running) and (time.time() - last_preview_at > PREVIEW_TTL_SEC):
        return Response(status=204)

    try:
        with proc.frame_lock:
            frame = proc.annotated_frame if proc.annotated_frame is not None else proc.current_frame
            if frame is None:
                return Response(status=204)
            frame = frame.copy()
    except Exception:
        return Response(status=204)

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return Response(status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/stream", methods=["GET"])
@require_login(api=True)
def stream_frames():
    """
    MJPEG 流预览，前端可直接用 <img src="/api/stream"> 显示。
    """
    def generate():
        while True:
            with processor_lock:
                proc = active_processor
            if not proc:
                time.sleep(0.05)
                continue

            try:
                with proc.frame_lock:
                    frame = proc.annotated_frame if proc.annotated_frame is not None else proc.current_frame
                    frame = frame.copy() if frame is not None else None
            except Exception:
                frame = None

            if frame is None:
                time.sleep(0.03)
                continue

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(0.03)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/probe", methods=["GET"])
@require_login(api=True)
def probe_camera():
    max_index = request.args.get("max_index", 20, type=int)
    devices = probe_camera_devices(max_index=max_index)
    if not devices:
        return jsonify(
            {
                "success": True,
                "data": {
                    "devices": [],
                    "recommended_index": None,
                    "message": "未检测到可用摄像头，请确认 EV 摄像头已连接且未被其他程序占用",
                },
            }
        )

    # 对课堂场景优先推荐非 0 号设备（通常 0 是笔记本内置摄像头）
    non_zero = [x for x in devices if x["index"] != 0]
    recommended = non_zero[0] if non_zero else devices[0]
    return jsonify(
        {
            "success": True,
            "data": {
                "devices": devices,
                "recommended_index": recommended["index"],
                "message": f"已检测到 {len(devices)} 个可用摄像头，推荐索引 {recommended['index']}",
            },
        }
    )


@app.route("/api/config/detection", methods=["POST"])
@require_login(api=True)
def update_detection_config():
    """
    动态更新检测阈值（让前端滑条真正影响模型推理）。
    说明：main.py 里使用的 DETECTION_CONFIG 是从 config.py 导入的 dict，
    这里直接 mutate `config.DETECTION_CONFIG`，运行中的推理会自动生效。
    """
    payload = request.get_json(silent=True) or {}
    conf = payload.get("conf_threshold")
    iou = payload.get("iou_threshold")
    update_interval = payload.get("update_interval")

    # 约束范围，避免无效配置
    if conf is not None:
        try:
            conf_f = float(conf)
            conf_f = max(0.01, min(0.99, conf_f))
            app_config.DETECTION_CONFIG["conf_threshold"] = conf_f
        except Exception:
            return jsonify({"success": False, "error": "conf_threshold 必须是数字"}), 400

    if iou is not None:
        try:
            iou_f = float(iou)
            iou_f = max(0.01, min(0.99, iou_f))
            app_config.DETECTION_CONFIG["iou_threshold"] = iou_f
        except Exception:
            return jsonify({"success": False, "error": "iou_threshold 必须是数字"}), 400

    if update_interval is not None:
        try:
            ui = float(update_interval)
            ui = max(0.01, min(2.0, ui))
            app_config.DETECTION_CONFIG["update_interval"] = ui
        except Exception:
            return jsonify({"success": False, "error": "update_interval 必须是数字"}), 400

    return jsonify(
        {
            "success": True,
            "data": {
                "conf_threshold": app_config.DETECTION_CONFIG.get("conf_threshold"),
                "iou_threshold": app_config.DETECTION_CONFIG.get("iou_threshold"),
                "update_interval": app_config.DETECTION_CONFIG.get("update_interval"),
            },
        }
    )


@app.route("/api/system/start", methods=["POST"])
@require_login(api=True)
def start_system():
    global video_thread, active_processor

    payload = request.get_json(silent=True) or {}
    camera = payload.get("camera")
    output = payload.get("output")
    registry_url = payload.get("registry_url")
    max_frames = payload.get("max_frames")
    model_name = payload.get("model_name")

    if not camera and not registry_url:
        return jsonify({"success": False, "error": "请提供 camera 或 registry_url"}), 400

    # 启动真实监控前：关闭 demo，并清空历史缓存，避免前端显示“固定的演示数字”
    with data_lock:
        detection_data["demo_mode"] = False
        detection_data["history"].clear()
        detection_data["alerts"].clear()
        detection_data["current"] = {"Person": 0, "Raise Hand": 0, "Lie Down": 0, "Phone Usage": 0}
        detection_data["last_update"] = datetime.now().isoformat()
        # 清除演示课程信息：让前端回退使用 last_update 显示“实时”
        # （index.html 优先展示 camera_info.class_time，一旦它是“周一08:00...”就会一直显示旧值）
        detection_data["camera_info"].update(
            {
                "course_name": "",
                "class_name": "",
                "teacher_name": "",
                "class_time": "",
                "room_id": "",
                "status": "starting",
            }
        )

    with video_thread_lock:
        # 如果线程还活着，但 processor 已经被清理了，说明上一轮刚结束/异常退出
        # 这种情况下允许重新启动（否则前端会一直提示“已在运行”）
        if video_thread and video_thread.is_alive():
            with processor_lock:
                proc = active_processor
            if proc is not None:
                return jsonify({"success": False, "error": "系统已在运行"}), 400
            video_thread = None

        # 禁用 VideoProcessor 内部重复启动 Flask（由当前 api_server.py 负责）
        os.environ["API_SERVER_START_ENABLED"] = "false"

        def worker():
            global active_processor, last_preview_at
            processor = None
            try:
                from main import (
                    VideoProcessor,
                    fetch_json_from_registry,
                    parse_json_data,
                    generate_output_filename,
                )

                # 识别输入为本地图片时，走单帧识别（VideoCapture 无法打开静态图片）
                image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                if camera and isinstance(camera, str) and os.path.exists(camera):
                    ext = os.path.splitext(camera)[1].lower()
                    if ext in image_exts:
                        processor = VideoProcessor(display_enabled=False, model_name=model_name)
                        with processor_lock:
                            active_processor = processor
                        img = cv2.imread(camera)
                        if img is None:
                            raise RuntimeError(f"图片读取失败: {camera}")
                        with processor.frame_lock:
                            processor.current_frame = img
                        processor.process_current_frame()
                        # 保留预览一段时间
                        last_preview_at = time.time()
                        return

                processor = VideoProcessor(display_enabled=False, model_name=model_name)
                with processor_lock:
                    active_processor = processor

                if camera:
                    output_filename = output or "output/demo.mp4"
                    processor.process_video(str(camera), output_filename, max_frames)
                elif registry_url:
                    json_data = fetch_json_from_registry(str(registry_url))
                    courses = parse_json_data(json_data)
                    for course in courses:
                        for device in course["devices"]:
                            if processor.should_stop:
                                break
                            output_filename = generate_output_filename(course, device)
                            processor.process_video(device["liveUrl"], output_filename, max_frames)
                            if processor.should_stop:
                                break
                else:
                    update_system_status({"running": False})
            except Exception as e:
                print(f"系统启动失败: {e}")
            finally:
                # 视频/流处理结束后清理 processor；图片单帧会在 TTL 内保留预览
                with processor_lock:
                    if active_processor is processor:
                        # 如果是图片单帧推理，保留结果供预览
                        if camera and isinstance(camera, str) and os.path.exists(camera):
                            ext = os.path.splitext(camera)[1].lower()
                            if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                                last_preview_at = time.time()
                                active_processor = processor
                            else:
                                active_processor = None
                        else:
                            active_processor = None
                with data_lock:
                    detection_data["camera_info"]["status"] = "stopped"
                update_system_status({"running": False})

        video_thread = threading.Thread(target=worker, daemon=True)
        video_thread.start()
        update_system_status({"running": True})
        return jsonify({"success": True, "message": "系统已启动"})


@app.route("/api/system/stop", methods=["POST"])
@require_login(api=True)
def stop_system():
    global video_thread, active_processor

    with video_thread_lock:
        with processor_lock:
            proc = active_processor
            if not proc:
                update_system_status({"running": False})
                with data_lock:
                    detection_data["camera_info"]["status"] = "stopped"
                return jsonify({"success": True, "message": "系统未运行"})

            proc.should_stop = True
            proc.should_stop_detection = True

        update_system_status({"running": False})
        with data_lock:
            detection_data["camera_info"]["status"] = "stopped"

        # 尽量等待线程退出，方便立刻重新开始
        t = video_thread
        if t:
            t.join(timeout=2.0)
            if not t.is_alive():
                video_thread = None

    return jsonify({"success": True, "message": "停止请求已发送"})


@app.route("/api/upload", methods=["POST"])
@require_login(api=True)
def upload_file():
    """
    上传图片/视频到服务器，返回服务器端可用路径，供 camera 输入使用。
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"success": False, "error": "未找到上传文件"}), 400

    # 简单清理文件名
    name = os.path.basename(f.filename).replace("..", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = UPLOAD_DIR / f"{ts}_{name}"
    f.save(str(save_path))
    return jsonify({"success": True, "path": str(save_path).replace("\\", "/")})


@app.route("/api/models", methods=["GET"])
@require_login(api=True)
def list_models():
    """
    返回可选模型列表：
    - models/ 下的本地 .pt
    - 一些 Ultralytics 预置模型名（会自动下载）
    """
    builtin = [
        "auto",
        "yolo11n.pt",
        "yolov8n.pt",
        "yolov8s.pt",
    ]
    local = sorted({p.name for p in MODEL_DIR.glob("*.pt")})
    return jsonify({"success": True, "data": {"builtin": builtin, "local": local}})


@app.route("/api/export/pdf", methods=["GET"])
@require_login(api=True)
def export_pdf():
    with data_lock:
        current = detection_data["current"].copy()
        camera_info = detection_data["camera_info"].copy()
        focus = detection_data["focus"].copy()
        history = list(detection_data["history"])[-20:]
        alerts = list(detection_data["alerts"])[-20:]
        status = detection_data["system_status"].copy()

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40

    def line(text, step=18):
        nonlocal y
        c.drawString(40, y, str(text))
        y -= step

    c.setFont("Helvetica-Bold", 14)
    line("Classroom Monitoring Report", 24)
    c.setFont("Helvetica", 10)
    line(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    line(f"Course: {camera_info.get('course_name', '-')}")
    line(f"Room: {camera_info.get('room_id', '-')}")
    line(f"Camera: {camera_info.get('name', '-')}")
    line(f"Running: {status.get('running')}  Frame Count: {status.get('frame_count')}  FPS: {status.get('fps')}", 22)

    c.setFont("Helvetica-Bold", 11)
    line("Current Counts:", 18)
    c.setFont("Helvetica", 10)
    line(f"Person={current.get('Person',0)}, Raise Hand={current.get('Raise Hand',0)}, Lie Down={current.get('Lie Down',0)}, Phone Usage={current.get('Phone Usage',0)}")
    line(f"Focus Score={focus.get('score', 80)}  Level={focus.get('level', '-')}", 22)

    c.setFont("Helvetica-Bold", 11)
    line("Recent History (latest 20):", 18)
    c.setFont("Helvetica", 9)
    if not history:
        line("No history data.", 16)
    else:
        for item in history:
            if y < 80:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 40
            counts = item.get("counts", {})
            line(
                f"{item.get('timestamp','-')} | P={counts.get('Person',0)} RH={counts.get('Raise Hand',0)} "
                f"LD={counts.get('Lie Down',0)} PU={counts.get('Phone Usage',0)} FS={item.get('focus',{}).get('score',80)}",
                14
            )

    if y < 120:
        c.showPage()
        y = height - 40
    c.setFont("Helvetica-Bold", 11)
    line("Recent Alerts (latest 20):", 18)
    c.setFont("Helvetica", 9)
    if not alerts:
        line("No alerts.", 16)
    else:
        for a in alerts:
            if y < 80:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 40
            d = a.get("details", {})
            line(f"{a.get('timestamp','-')} | LD={d.get('Lie Down',0)} PU={d.get('Phone Usage',0)}", 14)

    c.save()
    buf.seek(0)
    filename = f"monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": f"{SYSTEM_NAME} API",
            "timestamp": datetime.now().isoformat(),
        }
    )


def run_api_server(host="0.0.0.0", port=5000, debug=False):
    # 开启 threaded，避免 /api/stream 长连接阻塞其它接口
    init_auth_db()
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    run_api_server()
