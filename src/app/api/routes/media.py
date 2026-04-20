"""Media preview handlers extracted from legacy api_server."""

import time

import cv2
from flask import Response


def handle_get_latest_frame(
    processor_lock,
    get_active_processor,
    data_lock,
    detection_data,
    get_last_preview_at,
    preview_ttl_sec,
):
    """
    返回最新标注帧（JPEG）。
    前端轮询该接口实现“监控预览”。
    """
    with processor_lock:
        proc = get_active_processor()
    if not proc:
        return Response(status=204)
    # 仅在非运行态（图片单帧预览）应用 TTL；运行态视频流不限制
    with data_lock:
        running = bool(detection_data.get("system_status", {}).get("running", False))
    if (not running) and (time.time() - get_last_preview_at() > preview_ttl_sec):
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


def handle_stream_frames(processor_lock, get_active_processor):
    """
    MJPEG 流预览，前端可直接用 <img src="/api/stream"> 显示。
    """

    def generate():
        while True:
            with processor_lock:
                proc = get_active_processor()
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

