#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading
import signal
from datetime import datetime
from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import mediapipe as mp
import requests
import json
import argparse
from config import get_model_path, get_data_path, check_model_exists, DETECTION_CONFIG, PATH_CONFIG, API_CONFIG, SYSTEM_NAME

# 导入数据库模块
try:
    from database import get_db
    DB_ENABLED = True
except ImportError:
    print("警告: 数据库模块未找到，将仅使用内存存储")
    DB_ENABLED = False

# ---------------- 手机检测相关类 ----------------
class Autoencoder(nn.Module):
    """自编码器模型用于手机使用检测"""
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        
        super(Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        seq_len = x.size(1)
        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, seq_len, self.decoder.input_size, device=x.device)
        out, _ = self.decoder(decoder_input, (h_n, c_n))
        out = self.linear(out)
        return out

class SkeletonExtractor:
    """骨架提取器用于姿态分析"""
    def __init__(self, max_frames=300, num_joints=25, coords=3):
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.coords = coords
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_roi)
        data = np.zeros((self.max_frames, self.num_joints, self.coords))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            selected_indices = list(range(self.num_joints))
            for joint in range(min(self.num_joints, len(selected_indices))):
                lm = landmarks[selected_indices[joint]]
                data[0, joint, 0] = lm.x
                data[0, joint, 1] = lm.y
                data[0, joint, 2] = lm.z
            mean = np.mean(data[0], axis=0)
            std = np.std(data[0], axis=0) + 1e-8
            data[0] = (data[0] - mean) / std
            for frame_idx in range(1, self.max_frames):
                data[frame_idx] = data[0]
        data = data.reshape(self.max_frames, -1)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    def close(self):
        self.pose.close()

def detect_phone_usage(model, data, threshold, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """检测是否在玩手机"""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        error = torch.mean((data - output) ** 2).item()
    return error > threshold

def fetch_json_from_registry(registry_url):
    """从注册中心API获取JSON数据"""
    try:
        response = requests.get(registry_url)
        response.raise_for_status()
        json_data = response.json()
        print(f"Successfully fetched JSON from {registry_url}")
        return json_data
    except Exception as e:
        print(f"Error fetching JSON from registry: {e}")
        return []

def parse_json_data(json_data):
    """解析 JSON 数据并返回结构化信息"""
    courses = []
    try:
        for course in json_data:
            course_info = {
                "timeAdd": course.get("timeAdd"),
                "courseName": course.get("courseName"),
                "className": course.get("className") or course.get("class_name"),
                "teacherName": course.get("teacherName") or course.get("teacher_name"),
                "room_id": course.get("room_id", "unknown_room"),
                "devices": []
            }
            for device in course.get("device", []):
                device_info = {
                    "name": device.get("name"),
                    "type": device.get("type"),
                    "liveUrl": device.get("liveUrl"),
                    "model_number": device.get("model_number", ""),
                    "host": device.get("host", ""),
                    "serial_number": device.get("serial_number", ""),
                    "token": device.get("token", ""),
                    "admin": device.get("admin", ""),
                    "password": device.get("password", ""),
                    "create_by": device.get("create_by", ""),
                    "create_date": device.get("create_date", ""),
                    "update_by": device.get("update_by", ""),
                    "update_date": device.get("update_date", ""),
                    "status": device.get("status", ""),
                    "remarks": device.get("remarks", ""),
                    "del_flag": device.get("del_flag", "0")
                }
                course_info["devices"].append(device_info)
            courses.append(course_info)
        return courses
    except Exception as e:
        print(f"Error parsing JSON data: {e}")
        return []

def generate_output_filename(course_info, device_info):
    """生成输出文件名，格式：时间_课程_摄像头位置.mp4"""
    time_str = course_info["timeAdd"].replace(":", "-")  # 替换非法字符
    course_name = course_info["courseName"]
    camera_name = device_info["name"].replace("/", "_")  # 替换非法字符
    return f"{time_str}_{course_name}_{camera_name}.mp4"

class VideoProcessor:
    def __init__(self, display_enabled: bool = True, model_name: str | None = None):
        # 检测控制
        self.detection_enabled = True  # 默认启用检测
        self.model_name = (model_name or "").strip() or None
        
        # 检测结果（仅用于日志）
        self.detection_counts = {
            'Person': 0,
            'Raise Hand': 0,
            'Lie Down': 0,
            'Phone Usage': 0
        }
        
        # 时间相关
        self.last_update = time.time()
        self.update_interval = DETECTION_CONFIG['update_interval']  # 检测间隔
        self.last_detection_time = 0
        
        # 模型相关
        self.model = None
        self.is_processing = False
        
        # 手机检测相关
        self.phone_autoencoder = None
        self.skeleton_extractor = None
        self.phone_threshold = None
        self.phone_detection_enabled = True
        # 手机检测连续帧确认（需要连续2帧都检测到才认为是真的，减少误报）
        self.phone_detection_frames = {}  # 存储每个人的连续检测帧数 {bbox_key: frame_count}
        self.phone_confirm_frames = 2  # 需要连续检测到的帧数（降低到2帧以提高响应速度）
        
        # 线程控制
        self.detection_thread = None
        self.should_stop_detection = False
        
        # 录制相关
        self.output_video = None
        self.output_filename = None
        self.is_recording = False
        
        # 当前帧
        self.current_frame = None
        self.annotated_frame = None  # 带标注的帧，用于显示
        # 多线程共享帧：Web 端会读取 annotated_frame，因此需要锁保护
        self.frame_lock = threading.Lock()
        self.display_enabled = display_enabled
        
        # 停止标志
        self.should_stop = False
        
        # 显示窗口名称
        self.window_name = f"{SYSTEM_NAME} - 实时监控"
        
        # 摄像头信息（用于API）
        self.camera_info = {
            'name': '未知摄像头',
            'course_name': '未知课程',
            'room_id': '未知教室',
            'status': 'stopped'
        }
        
        # 检测统计（用于生成报告）
        self.detection_stats = {
            'total_frames': 0,
            'detected_frames': 0,  # 有检测结果的帧数
            'total_person': 0,
            'total_raise_hand': 0,
            'total_lie_down': 0,
            'total_phone_usage': 0,
            'max_person': 0,
            'max_raise_hand': 0,
            'max_lie_down': 0,
            'max_phone_usage': 0,
            'frames_with_lie_down': 0,  # 有躺下检测的帧数
            'frames_with_phone_usage': 0,  # 有手机使用的帧数
            'frames_with_raise_hand': 0,  # 有举手检测的帧数
        }
        self.start_time = None
        self.end_time = None
        
        # 初始化模型
        self.init_model()
        
        # 初始化手机检测模型
        self.init_phone_detection()
        
        # 初始化API服务
        self.init_api_server()
        
        # 初始化数据库
        self.db = None
        self.current_system_run_id = None
        if DB_ENABLED:
            try:
                self.db = get_db()
                print("数据库连接成功")
            except Exception as e:
                print(f"数据库连接失败: {e}")
                self.db = None
    
    def init_model(self):
        """初始化YOLO模型"""
        try:
            torch.set_float32_matmul_precision('medium')
            # 允许外部指定模型：可以是文件名/路径，也可以是 Ultralytics 支持的模型名（会自动下载）
            if self.model_name and self.model_name.lower() not in {"auto", "default"}:
                candidate = self.model_name
                # 若用户只填了文件名，优先在 model_path 下找
                if not os.path.isabs(candidate) and not os.path.exists(candidate):
                    candidate_in_dir = os.path.join(PATH_CONFIG["model_path"], candidate)
                    if os.path.exists(candidate_in_dir):
                        candidate = candidate_in_dir
                self.model = YOLO(candidate)
            else:
                # 自动选择：优先加载本地 best / yolov12n / yolo11n，否则用预置模型名（自动下载）
                model_path = get_model_path('best')
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                elif os.path.exists(get_model_path('yolov12n')):
                    self.model = YOLO(get_model_path('yolov12n'))
                elif os.path.exists(get_model_path('yolo11n')):
                    self.model = YOLO(get_model_path('yolo11n'))
                else:
                    self.model = YOLO('yolo11n.pt')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model.to(device=device, dtype=dtype)
            print(f"Model loaded on {device} (model={self.model_name or 'auto'})")
        except Exception as e:
            print(f"Model loading failed: {e}")
    
    def init_phone_detection(self):
        """初始化手机检测模型"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # 检查实际文件名
            phone_model_path = os.path.join(PATH_CONFIG['model_path'], 'phone_detection_autoencoder.pth')
            if os.path.exists(phone_model_path):
                self.phone_autoencoder = Autoencoder(25*3).to(device)
                self.phone_autoencoder.load_state_dict(torch.load(phone_model_path, map_location=device))
            else:
                print("Phone detection model not found, disabling")
                return
            threshold_path = os.path.join(PATH_CONFIG['model_path'], 'threshold.npy')
            if os.path.exists(threshold_path):
                self.phone_threshold = np.load(threshold_path)
            else:
                self.phone_threshold = 0.1
            self.skeleton_extractor = SkeletonExtractor()
            print("Phone detection initialized")
        except Exception as e:
            print(f"Phone detection init failed: {e}")
            self.phone_detection_enabled = False
    
    def init_api_server(self):
        """初始化API服务"""
        if API_CONFIG.get('enabled', False):
            try:
                from api_server import update_detection_data, update_system_status, run_api_server
                self.api_update_func = update_detection_data
                self.api_status_func = update_system_status
                
                # 如果已经由外部（例如 api_server.py）启动了 Flask 服务，则不重复启动
                api_server_start_enabled = os.getenv("API_SERVER_START_ENABLED", "true").lower() == "true"
                if api_server_start_enabled:
                    api_thread = threading.Thread(
                        target=run_api_server,
                        args=(API_CONFIG.get('host', '0.0.0.0'), API_CONFIG.get('port', 5000), False),
                        daemon=True
                    )
                    api_thread.start()
                    print(
                        f"API服务已启动: http://{API_CONFIG.get('host', '0.0.0.0')}:{API_CONFIG.get('port', 5000)}"
                    )
                else:
                    print("API服务启动被禁用：仅启用检测数据回调")
            except Exception as e:
                print(f"API服务启动失败: {e}")
                self.api_update_func = None
                self.api_status_func = None
        else:
            self.api_update_func = None
            self.api_status_func = None
    
    def load_video(self, video_source):
        """加载RTSP流或视频"""
        self.cap = None
        if video_source.startswith("rtsp://"):
            print(f"Loading RTSP: {video_source}")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        elif video_source.isdigit():
            # 如果是数字，作为摄像头索引
            camera_index = int(video_source)
            print(f"Loading camera: {camera_index}")
            self.cap = cv2.VideoCapture(camera_index)
        else:
            if not os.path.exists(video_source):
                print(f"File not found: {video_source}")
                return None
            print(f"Loading file: {video_source}")
            self.cap = cv2.VideoCapture(video_source)
        if self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = max(1, int(fps)) if fps > 0 else 30  # 如果无法获取FPS，默认30
            return self.cap
        else:
            print(f"Failed to load: {video_source}")
            return None
    
    def start_detection_thread(self):
        """启动检测线程"""
        def detection_worker():
            while not self.should_stop_detection:
                current_time = time.time()
                if self.detection_enabled and self.model and not self.is_processing and self.current_frame is not None and current_time - self.last_detection_time >= self.update_interval:
                    self.is_processing = True
                    try:
                        self.process_current_frame()
                        self.last_detection_time = current_time
                    except Exception as e:
                        print(f"Detection error: {e}")
                    finally:
                        self.is_processing = False
                time.sleep(0.05)
        self.detection_thread = threading.Thread(target=detection_worker, daemon=True)
        self.detection_thread.start()
    
    def stop_detection_thread(self):
        self.should_stop_detection = True
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
    
    def process_current_frame(self):
        """处理帧并标注四种检测"""
        with self.frame_lock:
            frame = self.current_frame
        if not self.model or frame is None:
            return
        results = self.model(
            frame,
            conf=DETECTION_CONFIG['conf_threshold'],
            verbose=False,
            max_det=DETECTION_CONFIG['max_det']
        )
        counts = {'Person': 0, 'Raise Hand': 0, 'Lie Down': 0, 'Phone Usage': 0}
        annotated_frame = frame.copy()
        lie_down_conf_threshold = DETECTION_CONFIG.get('lie_down_conf_threshold', 0.1)
        prone_aspect_ratio_threshold = DETECTION_CONFIG.get('prone_aspect_ratio_threshold', 1.3)

        # 从模型/结果中获取真实类别名，避免写死类别顺序导致“有画面但识别为 0”
        # Ultralytics YOLO: results[0].names / model.names 可能是 list 或 dict
        names = None
        try:
            if results and len(results) > 0 and getattr(results[0], "names", None) is not None:
                names = results[0].names
            elif getattr(self.model, "names", None) is not None:
                names = self.model.names
        except Exception:
            names = None

        def cls_name(cls_id: int) -> str:
            if names is None:
                return ""
            try:
                if isinstance(names, dict):
                    return str(names.get(int(cls_id), "") or "")
                if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
                    return str(names[int(cls_id)] or "")
            except Exception:
                return ""
            return ""

        def normalize_label(label: str) -> str:
            return (label or "").strip().lower().replace("_", " ")

        def map_behavior(label: str) -> str:
            """
            将模型输出标签映射到系统行为：
            - COCO 等通用模型至少能识别 person，用于 Person + 趴下(宽高比)推断
            - 自训模型若包含 Raise Hand / Lie Down 则直接使用
            """
            s = normalize_label(label)
            if not s:
                return ""
            if "person" in s:
                return "Person"
            if ("raise" in s and "hand" in s) or ("hand" in s and "up" in s):
                return "Raise Hand"
            if ("lie" in s and "down" in s) or "lying" in s or "sleep" in s or "prone" in s:
                return "Lie Down"
            return ""

        if results and results[0].boxes:
            for box in results[0].boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf)
                raw_label = cls_name(cls)
                behavior = map_behavior(raw_label)
                # fallback：如果模型没提供 names（极少），保留旧逻辑
                if not behavior:
                    class_names = ['Raise Hand', 'Person', 'Lie Down']
                    behavior = class_names[cls] if cls < len(class_names) else ""
                class_name = behavior if behavior else (raw_label or f'Class_{cls}')
                color = {
                    'Raise Hand': (0, 255, 0),
                    'Person': (0, 100, 255),
                    'Lie Down': (255, 0, 0),
                    'Phone Usage': (255, 165, 0)
                }.get(class_name, (128, 128, 128))
                # 按照优先级处理：Raise Hand > Lie Down > Person
                # 这样确保特殊动作优先于普通Person被识别
                if behavior == 'Raise Hand' and confidence >= 0.1:
                    # 举手检测
                    counts['Raise Hand'] += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif behavior == 'Lie Down' and confidence >= lie_down_conf_threshold:
                    # 睡觉检测：包括躺下和趴下
                    # YOLO模型已经将躺下和趴下都识别为"Lie Down"类别
                    # 这里不做额外区分，统一处理为睡觉状态
                    counts['Lie Down'] += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif behavior == 'Person' and confidence >= max(0.2, DETECTION_CONFIG.get('conf_threshold', 0.5)):
                    # Person检测：移除box_height限制，确保所有检测到的Person都被识别
                    # 包括：新人进来、离摄像头很近的人、从其他状态恢复的人等
                    # 只要置信度足够（>=0.5），就识别为Person
                    counts['Person'] += 1
                    bbox_width = max(1, x2 - x1)
                    bbox_height = max(1, y2 - y1)
                    aspect_ratio = bbox_width / bbox_height
                    is_prone_person = aspect_ratio >= prone_aspect_ratio_threshold
                    rect_color = color
                    label_text = f"{class_name}: {confidence:.2f}"
                    if is_prone_person:
                        counts['Lie Down'] += 1
                        rect_color = (255, 0, 0)
                        label_text = f"Lie Down (Prone): {confidence:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), rect_color, 2)
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
                
                # 手机检测：对所有检测到的Person（cls == 1）都进行检测，无论box_height大小
                # 这样可以检测到即使离摄像头很近的人（box_height >= 0.6）
                # 也确保新人进来时能检测到手机使用
                if behavior == 'Person' and confidence >= max(0.2, DETECTION_CONFIG.get('conf_threshold', 0.5)) and self.phone_detection_enabled and self.phone_autoencoder and self.skeleton_extractor:
                    # 生成唯一标识用于跟踪每个人
                    bbox_key = f"{x1}_{y1}_{x2}_{y2}"
                    skeleton_data = self.skeleton_extractor.extract(frame, (x1, y1, x2, y2))
                    if skeleton_data is not None:
                        is_using_phone = detect_phone_usage(self.phone_autoencoder, skeleton_data, self.phone_threshold)
                        if is_using_phone:
                            # 如果检测到手机使用，增加连续帧计数
                            if bbox_key in self.phone_detection_frames:
                                self.phone_detection_frames[bbox_key] += 1
                            else:
                                self.phone_detection_frames[bbox_key] = 1
                            
                            # 只有连续检测到足够帧数才认为是真的在使用手机
                            if self.phone_detection_frames[bbox_key] >= self.phone_confirm_frames:
                                counts['Phone Usage'] += 1
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 3)
                                cv2.putText(annotated_frame, f"Phone Usage: {confidence:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        else:
                            # 如果没有检测到手机使用，减少计数（而不是立即重置为0）
                            # 这样可以避免因为单帧检测失败而丢失连续检测
                            if bbox_key in self.phone_detection_frames:
                                if self.phone_detection_frames[bbox_key] > 0:
                                    self.phone_detection_frames[bbox_key] -= 1
                                if self.phone_detection_frames[bbox_key] <= 0:
                                    del self.phone_detection_frames[bbox_key]
                    
        
        # 清理手机检测帧记录（保留最近的检测结果，避免内存泄漏）
        if len(self.phone_detection_frames) > 50:
            # 只保留计数值大于0的记录
            self.phone_detection_frames = {k: v for k, v in self.phone_detection_frames.items() if v > 0}
        self.detection_counts = counts
        self.last_update = time.time()
        # 保存标注后的帧用于 Web 预览
        with self.frame_lock:
            self.annotated_frame = annotated_frame
        if self.is_recording:
            self.save_frame(annotated_frame)
        
        # 更新统计信息
        self.update_detection_stats(counts)
        
        # 更新API数据
        if self.api_update_func:
            try:
                self.api_update_func(counts, self.camera_info)
            except Exception as e:
                print(f"更新API数据失败: {e}")
        
        # 保存统计数据到数据库（定期保存，避免过于频繁）
        if hasattr(self, 'db') and self.db and self.detection_stats['total_frames'] % 100 == 0:
            try:
                if hasattr(self, 'current_system_run_id') and self.current_system_run_id:
                    self.db.create_or_update_detection_stats(
                        self.current_system_run_id,
                        self.detection_stats
                    )
            except Exception as e:
                print(f"保存统计数据到数据库失败: {e}")
        
        print(f"Frame processed: {counts}")
    
    def start_recording(self, output_filename, width, height, fps):
        os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        self.is_recording = True
        print(f"Recording to: {output_filename}")
    
    def stop_recording(self):
        if self.output_video:
            self.output_video.release()
            self.is_recording = False
            print("Recording stopped")
    
    def save_frame(self, frame):
        if self.is_recording and self.output_video:
            self.output_video.write(frame)
    
    def update_detection_stats(self, counts):
        """更新检测统计信息"""
        self.detection_stats['total_frames'] += 1
        
        # 检查是否有任何检测
        has_detection = any(counts.values())
        if has_detection:
            self.detection_stats['detected_frames'] += 1
        
        # 累计各类检测数量
        self.detection_stats['total_person'] += counts.get('Person', 0)
        self.detection_stats['total_raise_hand'] += counts.get('Raise Hand', 0)
        self.detection_stats['total_lie_down'] += counts.get('Lie Down', 0)
        self.detection_stats['total_phone_usage'] += counts.get('Phone Usage', 0)
        
        # 更新最大值
        self.detection_stats['max_person'] = max(self.detection_stats['max_person'], counts.get('Person', 0))
        self.detection_stats['max_raise_hand'] = max(self.detection_stats['max_raise_hand'], counts.get('Raise Hand', 0))
        self.detection_stats['max_lie_down'] = max(self.detection_stats['max_lie_down'], counts.get('Lie Down', 0))
        self.detection_stats['max_phone_usage'] = max(self.detection_stats['max_phone_usage'], counts.get('Phone Usage', 0))
        
        # 统计有异常行为的帧数
        if counts.get('Lie Down', 0) > 0:
            self.detection_stats['frames_with_lie_down'] += 1
        if counts.get('Phone Usage', 0) > 0:
            self.detection_stats['frames_with_phone_usage'] += 1
        if counts.get('Raise Hand', 0) > 0:
            self.detection_stats['frames_with_raise_hand'] += 1
    
    def generate_report(self, output_filename, frame_count, fps):
        """生成检测总结报告"""
        # 计算持续时间
        duration_seconds = 0
        if self.start_time and self.end_time:
            duration_seconds = (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            duration_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # 计算平均检测数量
        total_frames = self.detection_stats['total_frames']
        avg_person = self.detection_stats['total_person'] / total_frames if total_frames > 0 else 0
        avg_raise_hand = self.detection_stats['total_raise_hand'] / total_frames if total_frames > 0 else 0
        avg_lie_down = self.detection_stats['total_lie_down'] / total_frames if total_frames > 0 else 0
        avg_phone_usage = self.detection_stats['total_phone_usage'] / total_frames if total_frames > 0 else 0
        
        # 计算异常行为比例
        lie_down_ratio = (self.detection_stats['frames_with_lie_down'] / total_frames * 100) if total_frames > 0 else 0
        phone_usage_ratio = (self.detection_stats['frames_with_phone_usage'] / total_frames * 100) if total_frames > 0 else 0
        raise_hand_ratio = (self.detection_stats['frames_with_raise_hand'] / total_frames * 100) if total_frames > 0 else 0
        
        # 生成报告文件名
        report_filename = output_filename.replace('.mp4', '_报告.txt')
        if not report_filename.endswith('_报告.txt'):
            report_filename = output_filename + '_报告.txt'
        
        # 生成报告内容
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("课堂监控检测总结报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 基本信息
        report_lines.append("【基本信息】")
        report_lines.append(f"课程名称: {self.camera_info.get('course_name', '未知课程')}")
        report_lines.append(f"教室编号: {self.camera_info.get('room_id', '未知教室')}")
        report_lines.append(f"摄像头名称: {self.camera_info.get('name', '未知摄像头')}")
        report_lines.append(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else '未知'}")
        report_lines.append(f"结束时间: {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"持续时间: {duration_seconds:.1f} 秒 ({duration_seconds/60:.1f} 分钟)")
        report_lines.append(f"输出视频: {output_filename}")
        report_lines.append("")
        
        # 视频信息
        report_lines.append("【视频信息】")
        report_lines.append(f"总帧数: {frame_count}")
        report_lines.append(f"帧率: {fps} FPS")
        report_lines.append(f"视频时长: {frame_count/fps:.1f} 秒 ({frame_count/fps/60:.1f} 分钟)" if fps > 0 else "视频时长: 未知")
        report_lines.append("")
        
        # 检测统计
        report_lines.append("【检测统计】")
        report_lines.append(f"检测总帧数: {self.detection_stats['detected_frames']} / {total_frames} ({self.detection_stats['detected_frames']/total_frames*100:.1f}%)" if total_frames > 0 else "检测总帧数: 0")
        report_lines.append("")
        report_lines.append("人员检测:")
        report_lines.append(f"  - 累计检测次数: {self.detection_stats['total_person']}")
        report_lines.append(f"  - 平均每帧人数: {avg_person:.2f}")
        report_lines.append(f"  - 最大同时检测人数: {self.detection_stats['max_person']}")
        report_lines.append("")
        report_lines.append("举手检测:")
        report_lines.append(f"  - 累计检测次数: {self.detection_stats['total_raise_hand']}")
        report_lines.append(f"  - 平均每帧举手数: {avg_raise_hand:.2f}")
        report_lines.append(f"  - 最大同时举手数: {self.detection_stats['max_raise_hand']}")
        report_lines.append(f"  - 有举手行为的帧数: {self.detection_stats['frames_with_raise_hand']} ({raise_hand_ratio:.1f}%)")
        report_lines.append("")
        report_lines.append("躺下/睡觉检测:")
        report_lines.append(f"  - 累计检测次数: {self.detection_stats['total_lie_down']}")
        report_lines.append(f"  - 平均每帧躺下数: {avg_lie_down:.2f}")
        report_lines.append(f"  - 最大同时躺下数: {self.detection_stats['max_lie_down']}")
        report_lines.append(f"  - 有躺下行为的帧数: {self.detection_stats['frames_with_lie_down']} ({lie_down_ratio:.1f}%)")
        report_lines.append("")
        report_lines.append("手机使用检测:")
        report_lines.append(f"  - 累计检测次数: {self.detection_stats['total_phone_usage']}")
        report_lines.append(f"  - 平均每帧手机使用数: {avg_phone_usage:.2f}")
        report_lines.append(f"  - 最大同时手机使用数: {self.detection_stats['max_phone_usage']}")
        report_lines.append(f"  - 有手机使用的帧数: {self.detection_stats['frames_with_phone_usage']} ({phone_usage_ratio:.1f}%)")
        report_lines.append("")
        
        # 异常行为总结
        report_lines.append("【异常行为总结】")
        total_abnormal_frames = self.detection_stats['frames_with_lie_down'] + self.detection_stats['frames_with_phone_usage']
        abnormal_ratio = (total_abnormal_frames / total_frames * 100) if total_frames > 0 else 0
        report_lines.append(f"异常行为总帧数: {total_abnormal_frames} / {total_frames} ({abnormal_ratio:.1f}%)")
        report_lines.append("")
        if self.detection_stats['frames_with_lie_down'] > 0:
            report_lines.append(f"⚠️  检测到躺下/睡觉行为: {self.detection_stats['frames_with_lie_down']} 帧 ({lie_down_ratio:.1f}%)")
        if self.detection_stats['frames_with_phone_usage'] > 0:
            report_lines.append(f"⚠️  检测到手机使用行为: {self.detection_stats['frames_with_phone_usage']} 帧 ({phone_usage_ratio:.1f}%)")
        if total_abnormal_frames == 0:
            report_lines.append("✓ 未检测到异常行为")
        report_lines.append("")
        
        # 报告生成时间
        report_lines.append("=" * 80)
        report_lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        # 保存报告
        try:
            os.makedirs(os.path.dirname(report_filename) if os.path.dirname(report_filename) else '.', exist_ok=True)
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"\n{'='*60}")
            print(f"检测报告已生成: {report_filename}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"生成报告失败: {e}")
        
        # 同时在控制台输出报告摘要
        print("\n" + "=" * 60)
        print("检测总结报告摘要")
        print("=" * 60)
        print(f"课程: {self.camera_info.get('course_name', '未知课程')} | 教室: {self.camera_info.get('room_id', '未知教室')}")
        print(f"总帧数: {frame_count} | 持续时间: {duration_seconds:.1f}秒")
        print(f"人员检测: 累计{self.detection_stats['total_person']}次, 最大{self.detection_stats['max_person']}人")
        print(f"举手检测: 累计{self.detection_stats['total_raise_hand']}次, {self.detection_stats['frames_with_raise_hand']}帧有举手")
        print(f"躺下检测: 累计{self.detection_stats['total_lie_down']}次, {self.detection_stats['frames_with_lie_down']}帧有躺下")
        print(f"手机使用: 累计{self.detection_stats['total_phone_usage']}次, {self.detection_stats['frames_with_phone_usage']}帧有手机使用")
        print(f"异常行为比例: {abnormal_ratio:.1f}%")
        print("=" * 60 + "\n")
    
    def process_video(self, video_source, output_filename, max_frames=None, course_info=None, device_info=None):
        cap = self.load_video(video_source)
        if not cap:
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.fps
        self.start_recording(output_filename, width, height, fps)
        
        # 更新摄像头信息
        if course_info:
            self.camera_info['course_name'] = course_info.get('courseName', '未知课程')
            self.camera_info['class_name'] = course_info.get('className') or course_info.get('class_name') or '未知班级'
            self.camera_info['teacher_name'] = course_info.get('teacherName') or course_info.get('teacher_name') or '未知老师'
            self.camera_info['class_time'] = course_info.get('timeAdd', '未知时间')
            self.camera_info['room_id'] = course_info.get('room_id', '未知教室')
        if device_info:
            self.camera_info['name'] = device_info.get('name', '未知摄像头')
        self.camera_info['status'] = 'running'
        
        # 创建数据库运行记录
        if self.db:
            try:
                # 尝试查找或创建课程和设备记录
                course_id = None
                device_id = None
                
                if course_info:
                    course_name = course_info.get('courseName')
                    room_id = course_info.get('room_id')
                    if course_name and room_id:
                        # 查找现有课程
                        courses = self.db.get_courses()
                        for c in courses:
                            if c.get('course_name') == course_name and c.get('room_id') == room_id:
                                course_id = c.get('id')
                                break
                        # 如果不存在，创建新课程
                        if course_id is None:
                            course_id = self.db.create_course(
                                course_name=course_name,
                                room_id=room_id,
                                time_add=course_info.get('timeAdd'),
                                class_name=course_info.get('className') or course_info.get('class_name'),
                                teacher_name=course_info.get('teacherName') or course_info.get('teacher_name')
                            )
                
                if device_info:
                    device_name = device_info.get('name')
                    if device_name:
                        # 查找现有设备
                        # 这里简化处理，实际应该根据更多条件查找
                        device_id = self.db.create_device(
                            name=device_name,
                            course_id=course_id,
                            type=device_info.get('type'),
                            live_url=device_info.get('liveUrl'),
                            **{k: v for k, v in device_info.items() if k not in ['name', 'type', 'liveUrl']}
                        )
                
                # 创建系统运行记录
                self.current_system_run_id = self.db.create_system_run(
                    course_id=course_id,
                    device_id=device_id,
                    start_time=self.start_time,
                    status='running',
                    fps=fps
                )
                print(f"已创建数据库运行记录: ID={self.current_system_run_id}")
            except Exception as e:
                print(f"创建数据库运行记录失败: {e}")
        
        # 更新API中的摄像头信息
        if self.api_update_func:
            try:
                self.api_update_func(self.detection_counts, self.camera_info)
            except:
                pass
        
        # 更新系统状态
        if self.api_status_func:
            try:
                self.api_status_func({
                    'running': True,
                    'frame_count': 0,
                    'fps': fps
                })
            except:
                pass
        
        self.start_detection_thread()
        
        # 重置统计信息
        self.detection_stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'total_person': 0,
            'total_raise_hand': 0,
            'total_lie_down': 0,
            'total_phone_usage': 0,
            'max_person': 0,
            'max_raise_hand': 0,
            'max_lie_down': 0,
            'max_phone_usage': 0,
            'frames_with_lie_down': 0,
            'frames_with_phone_usage': 0,
            'frames_with_raise_hand': 0,
        }
        self.start_time = datetime.now()
        
        # 创建显示窗口（Web 模式一般不需要显示）
        if self.display_enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, width, height)
        
        frame_idx = 0
        print(f"\n{'='*60}")
        print("程序已开始运行，正在处理视频...")
        if max_frames:
            print(f"将处理最多 {max_frames} 帧")
        else:
            print("将一直运行，直到您按 'q' 键或 Ctrl+C 停止")
        print("提示：按 'q' 键退出，关闭窗口也会停止程序")
        print(f"{'='*60}\n")
        
        try:
            while not self.should_stop:
                ret, frame = cap.read()
                with self.frame_lock:
                    self.current_frame = frame if ret else None
                if not ret:
                    print("视频流结束，重新尝试连接...")
                    # 如果是摄像头，尝试重新打开
                    if video_source.isdigit():
                        cap.release()
                        time.sleep(1)
                        cap = self.load_video(video_source)
                        if not cap:
                            print("无法重新连接摄像头")
                            break
                    else:
                        break
                
                frame_idx += 1
                
                # 更新系统状态
                if self.api_status_func and frame_idx % 30 == 0:  # 每30帧更新一次
                    try:
                        self.api_status_func({
                            'running': True,
                            'frame_count': frame_idx,
                            'fps': fps
                        })
                    except:
                        pass
                
                if max_frames and frame_idx > max_frames:
                    print(f"\n已处理 {max_frames} 帧，停止处理")
                    break
                
                if self.display_enabled:
                    # 显示画面
                    with self.frame_lock:
                        annotated = self.annotated_frame
                        current = self.current_frame
                    display_frame = current.copy() if current is not None else None
                    if annotated is not None:
                        display_frame = annotated.copy()
                    
                    if display_frame is not None:
                        # 在画面上添加信息
                        info_text = [
                            f"Frame: {frame_idx}",
                            f"Person: {self.detection_counts['Person']}",
                            f"Raise Hand: {self.detection_counts['Raise Hand']}",
                            f"Lie Down: {self.detection_counts['Lie Down']}",
                            f"Phone Usage: {self.detection_counts['Phone Usage']}"
                        ]
                        y_offset = 30
                        for i, text in enumerate(info_text):
                            cv2.putText(
                                display_frame,
                                text,
                                (10, y_offset + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )
                        
                        # 显示画面
                        cv2.imshow(self.window_name, display_frame)
                        
                        # 检查窗口是否被关闭
                        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("\n窗口已关闭，停止程序...")
                            self.should_stop = True
                            break
                        
                        # 处理键盘输入（等待1ms，非阻塞）
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q'):
                            print("\n\n收到退出信号（按了 'q' 键），正在安全退出...")
                            self.should_stop = True
                            break
                
                # 每100帧显示一次进度
                if frame_idx % 100 == 0:
                    print(f"已处理 {frame_idx} 帧... (按 'q' 键或 Ctrl+C 可随时停止)")
                
                time.sleep(1.0 / fps)  # 模拟实时处理
        except KeyboardInterrupt:
            print("\n\n收到停止信号，正在安全退出...")
            self.should_stop = True
        finally:
            # 记录结束时间
            self.end_time = datetime.now()
            
            # 更新数据库运行记录
            if self.db and self.current_system_run_id:
                try:
                    duration_seconds = int((self.end_time - self.start_time).total_seconds()) if self.start_time else 0
                    self.db.update_system_run(
                        self.current_system_run_id,
                        end_time=self.end_time,
                        duration_seconds=duration_seconds,
                        total_frames=frame_idx,
                        status='completed',
                        output_video_path=output_filename,
                        report_path=output_filename.replace('.mp4', '_报告.txt') if output_filename else None
                    )
                    
                    # 保存最终统计数据
                    self.db.create_or_update_detection_stats(
                        self.current_system_run_id,
                        self.detection_stats
                    )
                    print(f"已更新数据库运行记录: ID={self.current_system_run_id}")
                except Exception as e:
                    print(f"更新数据库运行记录失败: {e}")
            
            # 关闭窗口
            if self.display_enabled:
                cv2.destroyAllWindows()
            self.stop_recording()
            self.stop_detection_thread()
            cap.release()
            if self.skeleton_extractor:
                self.skeleton_extractor.close()
            
            # 通知 Web 端：检测停止
            if self.api_status_func:
                try:
                    self.api_status_func({'running': False})
                except Exception:
                    pass
            
            print(f"\n处理完成！共处理 {frame_idx} 帧")
            print(f"输出文件: {output_filename}")
            
            # 生成检测报告
            if frame_idx > 0:
                self.generate_report(output_filename, frame_idx, fps)

def main():
    parser = argparse.ArgumentParser(description="视频处理脚本")
    parser.add_argument("--registry_url", type=str, default=None, help="注册中心URL，用于获取JSON（可选）")
    parser.add_argument("--camera", type=str, default=None, help="直接使用摄像头或视频源（摄像头索引如 '0'，或视频文件路径，或RTSP地址）")
    parser.add_argument("--output", type=str, default=None, help="输出文件名（使用--camera时必需）")
    parser.add_argument("--max_frames", type=int, default=None, help="最大处理帧数（默认无限，可一直运行直到按Ctrl+C停止）")
    parser.add_argument("--headless", action="store_true", help="不打开 OpenCV 展示窗口（适合 Web/服务器环境）")
    args = parser.parse_args()
    
    # 设置信号处理，优雅退出
    processor = None
    
    def signal_handler(sig, frame):
        print("\n\n正在停止程序...")
        if processor:
            processor.should_stop = True
        sys.exit(0)
    
    # 注册信号处理器（Windows 和 Linux 都支持）
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    processor = VideoProcessor(display_enabled=not args.headless)
    
    try:
        # 如果指定了 --camera，直接使用摄像头或视频源
        if args.camera:
            if not args.output:
                print("错误：使用 --camera 时，必须指定 --output 输出文件名")
                return
            print(f"使用摄像头/视频源: {args.camera}")
            processor.process_video(args.camera, args.output, args.max_frames)
        # 否则使用 registry_url 模式
        elif args.registry_url:
            json_data = fetch_json_from_registry(args.registry_url)
            courses = parse_json_data(json_data)
            for course in courses:
                for device in course["devices"]:
                    if processor.should_stop:
                        break
                    output_filename = generate_output_filename(course, device)
                    processor.process_video(device["liveUrl"], output_filename, args.max_frames)
                    if processor.should_stop:
                        break
        else:
            print("错误：请指定 --registry_url 或 --camera 参数")
            parser.print_help()
            return
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        if processor:
            processor.should_stop = True
    except Exception as e:
        print(f"\n发生错误: {e}")
        if processor:
            processor.should_stop = True

if __name__ == "__main__":
    main()