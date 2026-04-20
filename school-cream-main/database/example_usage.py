#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库使用示例
演示如何使用数据库模块进行数据查询和操作
"""
from database import get_db
from datetime import datetime, timedelta

def example_query_detection_records():
    """示例：查询检测记录"""
    print("=" * 60)
    print("示例1: 查询检测记录")
    print("=" * 60)
    
    db = get_db()
    
    # 查询最近的100条检测记录
    records = db.get_detection_records(limit=100)
    print(f"找到 {len(records)} 条检测记录")
    
    if records:
        print("\n最近的5条记录:")
        for i, record in enumerate(records[:5], 1):
            print(f"\n记录 {i}:")
            print(f"  时间: {record.get('timestamp')}")
            print(f"  人员: {record.get('person_count', 0)}")
            print(f"  举手: {record.get('raise_hand_count', 0)}")
            print(f"  躺下: {record.get('lie_down_count', 0)}")
            print(f"  手机使用: {record.get('phone_usage_count', 0)}")
            print(f"  课程: {record.get('course_name', '未知')}")
            print(f"  教室: {record.get('room_id', '未知')}")

def example_query_alerts():
    """示例：查询告警"""
    print("\n" + "=" * 60)
    print("示例2: 查询告警")
    print("=" * 60)
    
    db = get_db()
    
    # 查询未处理的告警
    alerts = db.get_alerts(is_resolved=False, limit=50)
    print(f"找到 {len(alerts)} 条未处理告警")
    
    if alerts:
        print("\n最近的5条告警:")
        for i, alert in enumerate(alerts[:5], 1):
            print(f"\n告警 {i}:")
            print(f"  类型: {alert.get('alert_type')}")
            print(f"  严重程度: {alert.get('severity')}")
            print(f"  躺下数量: {alert.get('lie_down_count', 0)}")
            print(f"  手机使用数量: {alert.get('phone_usage_count', 0)}")
            print(f"  时间: {alert.get('created_at')}")
            print(f"  课程: {alert.get('course_name', '未知')}")

def example_query_stats():
    """示例：查询统计数据"""
    print("\n" + "=" * 60)
    print("示例3: 查询统计数据")
    print("=" * 60)
    
    db = get_db()
    
    # 获取系统运行记录
    # 这里需要先有一个system_run_id，实际使用时从system_runs表获取
    # 示例：假设system_run_id=1
    stats = db.get_detection_stats(1)
    
    if stats:
        print("\n检测统计信息:")
        print(f"  总帧数: {stats.get('total_frames', 0)}")
        print(f"  检测帧数: {stats.get('detected_frames', 0)}")
        print(f"  累计人员检测: {stats.get('total_person', 0)}")
        print(f"  累计举手检测: {stats.get('total_raise_hand', 0)}")
        print(f"  累计躺下检测: {stats.get('total_lie_down', 0)}")
        print(f"  累计手机使用: {stats.get('total_phone_usage', 0)}")
        print(f"  最大同时人数: {stats.get('max_person', 0)}")
        print(f"  躺下行为比例: {stats.get('lie_down_ratio', 0):.2f}%")
        print(f"  手机使用比例: {stats.get('phone_usage_ratio', 0):.2f}%")
        print(f"  异常行为比例: {stats.get('abnormal_ratio', 0):.2f}%")
    else:
        print("未找到统计数据（可能需要先运行系统生成数据）")

def example_create_course():
    """示例：创建课程"""
    print("\n" + "=" * 60)
    print("示例4: 创建课程")
    print("=" * 60)
    
    db = get_db()
    
    # 创建课程
    course_id = db.create_course(
        course_name="高等数学",
        room_id="A101",
        time_add="14:30:00",
        status="active",
        remarks="示例课程"
    )
    print(f"已创建课程，ID: {course_id}")

def example_resolve_alert():
    """示例：处理告警"""
    print("\n" + "=" * 60)
    print("示例5: 处理告警")
    print("=" * 60)
    
    db = get_db()
    
    # 获取未处理的告警
    alerts = db.get_alerts(is_resolved=False, limit=1)
    
    if alerts:
        alert_id = alerts[0].get('id')
        print(f"处理告警 ID: {alert_id}")
        db.resolve_alert(alert_id, resolved_by="管理员")
        print("告警已标记为已处理")
    else:
        print("没有未处理的告警")

def example_query_by_time_range():
    """示例：按时间范围查询"""
    print("\n" + "=" * 60)
    print("示例6: 按时间范围查询")
    print("=" * 60)
    
    db = get_db()
    
    # 查询最近1小时的记录
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    records = db.get_detection_records(
        start_time=start_time,
        end_time=end_time,
        limit=100
    )
    
    print(f"最近1小时内有 {len(records)} 条检测记录")

if __name__ == '__main__':
    try:
        # 运行示例
        example_query_detection_records()
        example_query_alerts()
        example_query_stats()
        # example_create_course()  # 取消注释以创建示例课程
        # example_resolve_alert()  # 取消注释以处理告警
        example_query_by_time_range()
        
        print("\n" + "=" * 60)
        print("示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭数据库连接
        db = get_db()
        db.close()

