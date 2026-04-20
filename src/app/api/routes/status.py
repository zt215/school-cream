"""Status-related handlers extracted from legacy api_server."""

from flask import jsonify, request


def handle_get_status(detection_data, data_lock, seed_demo_data):
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


def handle_get_history(detection_data, data_lock):
    limit = request.args.get("limit", 100, type=int)
    with data_lock:
        items = list(detection_data["history"])[-limit:]
    return jsonify({"success": True, "data": items, "total": len(items)})


def handle_get_alerts(detection_data, data_lock):
    limit = request.args.get("limit", 50, type=int)
    with data_lock:
        items = list(detection_data["alerts"])[-limit:]
    return jsonify({"success": True, "data": items, "total": len(items)})


def handle_get_stats(detection_data, data_lock):
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


def handle_focus_realtime(detection_data, data_lock):
    with data_lock:
        focus = detection_data["focus"]
        modal = detection_data["multimodal"]
    return jsonify({"success": True, "data": {"focus": focus, "multimodal": modal}})

