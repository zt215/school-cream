"""Upload/model/export handlers extracted from legacy api_server."""

import os
from datetime import datetime
from io import BytesIO

from flask import jsonify, request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def handle_upload_file(upload_dir):
    """
    上传图片/视频到服务器，返回服务器端可用路径，供 camera 输入使用。
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"success": False, "error": "未找到上传文件"}), 400

    # 简单清理文件名
    name = os.path.basename(f.filename).replace("..", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = upload_dir / f"{ts}_{name}"
    f.save(str(save_path))
    return jsonify({"success": True, "path": str(save_path).replace("\\", "/")})


def handle_list_models(model_dir):
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
    local = sorted({p.name for p in model_dir.glob("*.pt")})
    return jsonify({"success": True, "data": {"builtin": builtin, "local": local}})


def handle_export_pdf(detection_data, data_lock):
    with data_lock:
        current = detection_data["current"].copy()
        camera_info = detection_data["camera_info"].copy()
        focus = detection_data["focus"].copy()
        history = list(detection_data["history"])[-20:]
        alerts = list(detection_data["alerts"])[-20:]
        status = detection_data["system_status"].copy()

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _width, height = A4
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

