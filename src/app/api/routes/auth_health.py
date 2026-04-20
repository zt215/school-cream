"""Auth and health handlers extracted from legacy api_server."""

from datetime import datetime
import sqlite3

from flask import jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


def handle_login(auth_db):
    if request.method == "GET":
        if session.get("user_id"):
            return redirect(url_for("index_page"))
        return render_template("login.html", error="")

    payload = request.form or (request.get_json(silent=True) or {})
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    if not username or not password:
        return render_template("login.html", error="用户名和密码不能为空"), 400

    conn = sqlite3.connect(str(auth_db))
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


def handle_register(auth_db):
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

    conn = sqlite3.connect(str(auth_db))
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


def handle_logout():
    session.clear()
    return redirect(url_for("login_page"))


def handle_health(system_name: str):
    return jsonify(
        {
            "status": "healthy",
            "service": f"{system_name} API",
            "timestamp": datetime.now().isoformat(),
        }
    )

