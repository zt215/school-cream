"""System start/stop handlers extracted from legacy api_server."""

import os
import threading
import time

import cv2
from flask import jsonify, request


def handle_start_system(
    *,
    video_thread_lock,
    processor_lock,
    data_lock,
    detection_data,
    get_video_thread,
    set_video_thread,
    get_active_processor,
    set_active_processor,
    set_last_preview_at,
    update_system_status,
):
    payload = request.get_json(silent=True) or {}
    camera = payload.get("camera")
    output = payload.get("output")
    registry_url = payload.get("registry_url")
    max_frames = payload.get("max_frames")
    model_name = payload.get("model_name")

    if not camera and not registry_url:
        return jsonify({"success": False, "error": "请提供 camera 或 registry_url"}), 400

    with video_thread_lock:
        # 如果线程还活着，但 processor 已经被清理了，说明上一轮刚结束/异常退出
        # 这种情况下允许重新启动（否则前端会一直提示“已在运行”）
        video_thread = get_video_thread()
        if video_thread and video_thread.is_alive():
            with processor_lock:
                proc = get_active_processor()
            if proc is not None:
                return jsonify({"success": False, "error": "系统已在运行"}), 400
            set_video_thread(None)

        # 禁用 VideoProcessor 内部重复启动 Flask（由当前 api_server.py 负责）
        os.environ["API_SERVER_START_ENABLED"] = "false"

        def worker():
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
                            set_active_processor(processor)
                        img = cv2.imread(camera)
                        if img is None:
                            raise RuntimeError(f"图片读取失败: {camera}")
                        with processor.frame_lock:
                            processor.current_frame = img
                        processor.process_current_frame()
                        # 保留预览一段时间
                        set_last_preview_at(time.time())
                        return

                processor = VideoProcessor(display_enabled=False, model_name=model_name)
                with processor_lock:
                    set_active_processor(processor)

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
                    if get_active_processor() is processor:
                        # 如果是图片单帧推理，保留结果供预览
                        if camera and isinstance(camera, str) and os.path.exists(camera):
                            ext = os.path.splitext(camera)[1].lower()
                            if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                                set_last_preview_at(time.time())
                                set_active_processor(processor)
                            else:
                                set_active_processor(None)
                        else:
                            set_active_processor(None)
                with data_lock:
                    detection_data["camera_info"]["status"] = "stopped"
                update_system_status({"running": False})

        video_thread = threading.Thread(target=worker, daemon=True)
        set_video_thread(video_thread)
        video_thread.start()
        update_system_status({"running": True})
        return jsonify({"success": True, "message": "系统已启动"})


def handle_stop_system(
    *,
    video_thread_lock,
    processor_lock,
    data_lock,
    detection_data,
    get_video_thread,
    set_video_thread,
    get_active_processor,
    update_system_status,
):
    with video_thread_lock:
        with processor_lock:
            proc = get_active_processor()
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
        t = get_video_thread()
        if t:
            t.join(timeout=2.0)
            if not t.is_alive():
                set_video_thread(None)

    return jsonify({"success": True, "message": "停止请求已发送"})

