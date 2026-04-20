#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库操作模块
支持 SQLite, MySQL, PostgreSQL
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, List, Any
import json
import threading

# 尝试导入其他数据库驱动
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

# 数据库配置
DB_CONFIG = {
    'type': os.getenv('DB_TYPE', 'sqlite').lower(),  # sqlite, mysql, postgresql
    'sqlite_path': os.getenv('DB_PATH', './data/school_cream.db'),
    'mysql_host': os.getenv('MYSQL_HOST', 'localhost'),
    'mysql_port': int(os.getenv('MYSQL_PORT', '3306')),
    'mysql_user': os.getenv('MYSQL_USER', 'root'),
    'mysql_password': os.getenv('MYSQL_PASSWORD', ''),
    'mysql_database': os.getenv('MYSQL_DATABASE', 'school_cream'),
    'postgresql_host': os.getenv('POSTGRESQL_HOST', 'localhost'),
    'postgresql_port': int(os.getenv('POSTGRESQL_PORT', '5432')),
    'postgresql_user': os.getenv('POSTGRESQL_USER', 'postgres'),
    'postgresql_password': os.getenv('POSTGRESQL_PASSWORD', ''),
    'postgresql_database': os.getenv('POSTGRESQL_DATABASE', 'school_cream'),
}

# 线程锁
db_lock = threading.Lock()


class Database:
    """数据库操作类"""
    
    def __init__(self):
        self.conn = None
        self.db_type = DB_CONFIG['type']
        self._connect()
        self._init_database()
    
    def _connect(self):
        """连接数据库"""
        try:
            if self.db_type == 'sqlite':
                # 确保目录存在
                db_path = DB_CONFIG['sqlite_path']
                os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
                self.conn = sqlite3.connect(db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row
                print(f"已连接到 SQLite 数据库: {db_path}")
            elif self.db_type == 'mysql':
                if not MYSQL_AVAILABLE:
                    raise ImportError("pymysql 未安装，请运行: pip install pymysql")
                self.conn = pymysql.connect(
                    host=DB_CONFIG['mysql_host'],
                    port=DB_CONFIG['mysql_port'],
                    user=DB_CONFIG['mysql_user'],
                    password=DB_CONFIG['mysql_password'],
                    database=DB_CONFIG['mysql_database'],
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
                print(f"已连接到 MySQL 数据库: {DB_CONFIG['mysql_database']}")
            elif self.db_type == 'postgresql':
                if not POSTGRESQL_AVAILABLE:
                    raise ImportError("psycopg2 未安装，请运行: pip install psycopg2-binary")
                self.conn = psycopg2.connect(
                    host=DB_CONFIG['postgresql_host'],
                    port=DB_CONFIG['postgresql_port'],
                    user=DB_CONFIG['postgresql_user'],
                    password=DB_CONFIG['postgresql_password'],
                    database=DB_CONFIG['postgresql_database']
                )
                self.conn.set_session(autocommit=False)
                print(f"已连接到 PostgreSQL 数据库: {DB_CONFIG['postgresql_database']}")
            else:
                raise ValueError(f"不支持的数据库类型: {self.db_type}")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise
    
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            schema_file = None
            if self.db_type == 'sqlite':
                schema_file = 'database/schema.sql'
            elif self.db_type == 'mysql':
                schema_file = 'database/schema_mysql.sql'
            elif self.db_type == 'postgresql':
                schema_file = 'database/schema_postgresql.sql'
            
            if schema_file and os.path.exists(schema_file):
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                
                # 执行SQL脚本
                cursor = self.conn.cursor()
                # 对于SQLite，需要逐条执行
                if self.db_type == 'sqlite':
                    for statement in schema_sql.split(';'):
                        statement = statement.strip()
                        if statement and not statement.startswith('--'):
                            try:
                                cursor.execute(statement)
                            except sqlite3.OperationalError as e:
                                # 忽略已存在的表/索引错误
                                if 'already exists' not in str(e).lower():
                                    print(f"执行SQL时出错: {e}")
                else:
                    cursor.execute(schema_sql)
                self.conn.commit()
                cursor.close()
                print("数据库表结构初始化完成")
                self._ensure_course_columns()
                self._seed_default_courses_if_empty()
        except Exception as e:
            print(f"初始化数据库表结构失败: {e}")

    def _ensure_course_columns(self):
        """为已有库补齐新课程字段（班级、授课老师）"""
        alter_statements = []
        if self.db_type == 'sqlite':
            alter_statements = [
                "ALTER TABLE courses ADD COLUMN class_name VARCHAR(100)",
                "ALTER TABLE courses ADD COLUMN teacher_name VARCHAR(100)",
            ]
        elif self.db_type == 'mysql':
            alter_statements = [
                "ALTER TABLE courses ADD COLUMN class_name VARCHAR(100) COMMENT '班级名称'",
                "ALTER TABLE courses ADD COLUMN teacher_name VARCHAR(100) COMMENT '授课老师'",
            ]
        elif self.db_type == 'postgresql':
            alter_statements = [
                "ALTER TABLE courses ADD COLUMN class_name VARCHAR(100)",
                "ALTER TABLE courses ADD COLUMN teacher_name VARCHAR(100)",
            ]

        for sql in alter_statements:
            try:
                self._execute(sql)
            except Exception as e:
                msg = str(e).lower()
                if "duplicate column" in msg or "already exists" in msg:
                    continue
                print(f"补齐课程字段失败: {e}")

    def _seed_default_courses_if_empty(self):
        """空库时预置示例课程（来自课表示例）"""
        try:
            row = self._fetchone("SELECT COUNT(1) AS cnt FROM courses")
            if row and int(row.get("cnt", 0)) > 0:
                return
        except Exception as e:
            print(f"检测课程数量失败，跳过预置: {e}")
            return

        sample_courses = [
            {
                "course_name": "软件质量保证与测试",
                "room_id": "C-219",
                "time_add": "周一 08:00-09:50（1-2节）",
                "class_name": "2023级 软件工程1班",
                "teacher_name": "冯欣",
            },
            {
                "course_name": "计算机机房建设与维护",
                "room_id": "A-201",
                "time_add": "周二 10:10-12:00（3-4节）",
                "class_name": "2023级 软件工程1班",
                "teacher_name": "杨书峰",
            },
            {
                "course_name": "软件工程综合实践",
                "room_id": "A-503",
                "time_add": "周三 08:00-09:50（1-2节）",
                "class_name": "2023级 软件工程1班",
                "teacher_name": "戴海滨",
            },
            {
                "course_name": "就业指导与职业发展",
                "room_id": "A-509",
                "time_add": "周三 14:30-16:20（5-6节）",
                "class_name": "2023级 软件工程1班",
                "teacher_name": "侯辰",
            },
            {
                "course_name": "Python程序设计语言",
                "room_id": "A-205",
                "time_add": "周四 14:30-17:30（5-8节）",
                "class_name": "2023级 软件工程1班",
                "teacher_name": "张宏琳",
            },
        ]
        for item in sample_courses:
            try:
                self.create_course(**item)
            except Exception as e:
                print(f"预置课程写入失败: {item.get('course_name')} -> {e}")
    
    def _execute(self, sql: str, params: tuple = None):
        """执行SQL语句"""
        with db_lock:
            try:
                cursor = self.conn.cursor()
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                self.conn.commit()
                return cursor
            except Exception as e:
                self.conn.rollback()
                print(f"执行SQL失败: {e}\nSQL: {sql}\nParams: {params}")
                raise
    
    def _fetchall(self, sql: str, params: tuple = None) -> List[Dict]:
        """查询多条记录"""
        cursor = self._execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        
        # 转换为字典列表
        if self.db_type == 'sqlite':
            return [dict(row) for row in rows]
        else:
            return rows
    
    def _fetchone(self, sql: str, params: tuple = None) -> Optional[Dict]:
        """查询单条记录"""
        cursor = self._execute(sql, params)
        row = cursor.fetchone()
        cursor.close()
        
        if row is None:
            return None
        
        # 转换为字典
        if self.db_type == 'sqlite':
            return dict(row)
        else:
            return row
    
    # ==================== 课程相关操作 ====================
    
    def create_course(self, course_name: str, room_id: str, time_add: str = None,
                     class_name: str = None, teacher_name: str = None,
                     status: str = 'active', remarks: str = None) -> int:
        """创建课程"""
        sql = """
            INSERT INTO courses (course_name, room_id, time_add, class_name, teacher_name, status, remarks)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        cursor = self._execute(sql, (course_name, room_id, time_add, class_name, teacher_name, status, remarks))
        if self.db_type == 'sqlite':
            return cursor.lastrowid
        elif self.db_type == 'mysql':
            return cursor.lastrowid
        else:  # postgresql
            return cursor.fetchone()[0]
    
    def get_course(self, course_id: int) -> Optional[Dict]:
        """获取课程信息"""
        sql = "SELECT * FROM courses WHERE id = ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        return self._fetchone(sql, (course_id,))
    
    def get_courses(self, status: str = None) -> List[Dict]:
        """获取课程列表"""
        if status:
            sql = "SELECT * FROM courses WHERE status = ? ORDER BY created_at DESC"
            if self.db_type == 'postgresql':
                sql = sql.replace('?', '%s')
            return self._fetchall(sql, (status,))
        else:
            sql = "SELECT * FROM courses ORDER BY created_at DESC"
            return self._fetchall(sql)
    
    # ==================== 设备相关操作 ====================
    
    def create_device(self, name: str, course_id: int = None, **kwargs) -> int:
        """创建设备"""
        fields = ['name', 'course_id', 'type', 'live_url', 'model_number', 
                 'host', 'serial_number', 'token', 'admin', 'password', 
                 'status', 'create_by', 'create_date', 'update_by', 
                 'update_date', 'del_flag', 'remarks']
        values = []
        placeholders = []
        
        for field in fields:
            if field in kwargs or field == 'name' or field == 'course_id':
                placeholders.append('?')
                if field == 'name':
                    values.append(name)
                elif field == 'course_id':
                    values.append(course_id)
                else:
                    values.append(kwargs.get(field))
        
        sql = f"""
            INSERT INTO devices ({', '.join([f for f in fields if f in kwargs or f == 'name' or f == 'course_id'])})
            VALUES ({', '.join(placeholders)})
        """
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        
        cursor = self._execute(sql, tuple(values))
        if self.db_type == 'sqlite':
            return cursor.lastrowid
        elif self.db_type == 'mysql':
            return cursor.lastrowid
        else:  # postgresql
            return cursor.fetchone()[0]
    
    def get_device(self, device_id: int) -> Optional[Dict]:
        """获取设备信息"""
        sql = "SELECT * FROM devices WHERE id = ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        return self._fetchone(sql, (device_id,))
    
    # ==================== 系统运行相关操作 ====================
    
    def create_system_run(self, course_id: int = None, device_id: int = None,
                         start_time: datetime = None, **kwargs) -> int:
        """创建系统运行记录"""
        if start_time is None:
            start_time = datetime.now()
        
        sql = """
            INSERT INTO system_runs (course_id, device_id, start_time, status, command, pid, fps)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        
        status = kwargs.get('status', 'running')
        command = kwargs.get('command')
        pid = kwargs.get('pid')
        fps = kwargs.get('fps')
        
        cursor = self._execute(sql, (course_id, device_id, start_time, status, command, pid, fps))
        if self.db_type == 'sqlite':
            return cursor.lastrowid
        elif self.db_type == 'mysql':
            return cursor.lastrowid
        else:  # postgresql
            return cursor.fetchone()[0]
    
    def update_system_run(self, run_id: int, **kwargs):
        """更新系统运行记录"""
        updates = []
        values = []
        for key, value in kwargs.items():
            updates.append(f"{key} = ?")
            values.append(value)
        
        if not updates:
            return
        
        values.append(run_id)
        sql = f"UPDATE system_runs SET {', '.join(updates)} WHERE id = ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        
        self._execute(sql, tuple(values))
    
    def get_system_run(self, run_id: int) -> Optional[Dict]:
        """获取系统运行记录"""
        sql = "SELECT * FROM system_runs WHERE id = ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        return self._fetchone(sql, (run_id,))
    
    # ==================== 检测记录相关操作 ====================
    
    def create_detection_record(self, system_run_id: int, counts: Dict,
                              frame_number: int = None, timestamp: datetime = None,
                              camera_info: Dict = None) -> int:
        """创建检测记录"""
        if timestamp is None:
            timestamp = datetime.now()
        
        sql = """
            INSERT INTO detection_records 
            (system_run_id, frame_number, timestamp, person_count, raise_hand_count, 
             lie_down_count, phone_usage_count, camera_name, course_name, room_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        
        camera_name = camera_info.get('name') if camera_info else None
        course_name = camera_info.get('course_name') if camera_info else None
        room_id = camera_info.get('room_id') if camera_info else None
        
        cursor = self._execute(sql, (
            system_run_id,
            frame_number,
            timestamp,
            counts.get('Person', 0),
            counts.get('Raise Hand', 0),
            counts.get('Lie Down', 0),
            counts.get('Phone Usage', 0),
            camera_name,
            course_name,
            room_id
        ))
        
        if self.db_type == 'sqlite':
            return cursor.lastrowid
        elif self.db_type == 'mysql':
            return cursor.lastrowid
        else:  # postgresql
            return cursor.fetchone()[0]
    
    def get_detection_records(self, system_run_id: int = None, limit: int = 100,
                             start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """获取检测记录"""
        conditions = []
        params = []
        
        if system_run_id:
            conditions.append("system_run_id = ?")
            params.append(system_run_id)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM detection_records {where_clause} ORDER BY timestamp DESC LIMIT ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        else:
            params.append(limit)
        
        return self._fetchall(sql, tuple(params))
    
    # ==================== 告警相关操作 ====================
    
    def create_alert(self, alert_type: str, system_run_id: int = None,
                    detection_record_id: int = None, counts: Dict = None,
                    camera_info: Dict = None, severity: str = 'medium') -> int:
        """创建告警记录"""
        sql = """
            INSERT INTO alerts 
            (system_run_id, detection_record_id, alert_type, severity, 
             lie_down_count, phone_usage_count, camera_name, course_name, room_id, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        
        camera_name = camera_info.get('name') if camera_info else None
        course_name = camera_info.get('course_name') if camera_info else None
        room_id = camera_info.get('room_id') if camera_info else None
        details = json.dumps(counts) if counts else None
        
        cursor = self._execute(sql, (
            system_run_id,
            detection_record_id,
            alert_type,
            severity,
            counts.get('Lie Down', 0) if counts else 0,
            counts.get('Phone Usage', 0) if counts else 0,
            camera_name,
            course_name,
            room_id,
            details
        ))
        
        if self.db_type == 'sqlite':
            return cursor.lastrowid
        elif self.db_type == 'mysql':
            return cursor.lastrowid
        else:  # postgresql
            return cursor.fetchone()[0]
    
    def get_alerts(self, is_resolved: bool = None, limit: int = 100) -> List[Dict]:
        """获取告警列表"""
        if is_resolved is not None:
            sql = "SELECT * FROM alerts WHERE is_resolved = ? ORDER BY created_at DESC LIMIT ?"
            if self.db_type == 'postgresql':
                sql = sql.replace('?', '%s')
            return self._fetchall(sql, (is_resolved, limit))
        else:
            sql = "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?"
            if self.db_type == 'postgresql':
                sql = sql.replace('?', '%s')
            return self._fetchall(sql, (limit,))
    
    def resolve_alert(self, alert_id: int, resolved_by: str = None):
        """处理告警"""
        sql = "UPDATE alerts SET is_resolved = 1, resolved_at = ?, resolved_by = ? WHERE id = ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        self._execute(sql, (datetime.now(), resolved_by, alert_id))
    
    # ==================== 统计相关操作 ====================
    
    def create_or_update_detection_stats(self, system_run_id: int, stats: Dict):
        """创建或更新检测统计"""
        # 检查是否已存在
        sql = "SELECT id FROM detection_stats WHERE system_run_id = ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        existing = self._fetchone(sql, (system_run_id,))
        
        if existing:
            # 更新
            updates = []
            values = []
            for key, value in stats.items():
                updates.append(f"{key} = ?")
                values.append(value)
            values.append(system_run_id)
            sql = f"UPDATE detection_stats SET {', '.join(updates)}, updated_at = ? WHERE system_run_id = ?"
            if self.db_type == 'postgresql':
                sql = sql.replace('?', '%s')
            values.insert(-1, datetime.now())
            self._execute(sql, tuple(values))
        else:
            # 创建
            fields = list(stats.keys()) + ['system_run_id']
            placeholders = ['?' for _ in fields]
            values = list(stats.values()) + [system_run_id]
            
            sql = f"""
                INSERT INTO detection_stats ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
            """
            if self.db_type == 'postgresql':
                sql = sql.replace('?', '%s')
            self._execute(sql, tuple(values))
    
    def get_detection_stats(self, system_run_id: int) -> Optional[Dict]:
        """获取检测统计"""
        sql = "SELECT * FROM detection_stats WHERE system_run_id = ?"
        if self.db_type == 'postgresql':
            sql = sql.replace('?', '%s')
        return self._fetchone(sql, (system_run_id,))
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("数据库连接已关闭")


# 全局数据库实例
_db_instance = None


def get_db() -> Database:
    """获取数据库实例（单例模式）"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


if __name__ == '__main__':
    # 测试数据库连接
    db = get_db()
    print("数据库连接测试成功！")
    db.close()

