# school-cream
青衿智眼

## 模型说明（assets 里的模型有什么用）
- 前端会调用 `/api/models`（`assets` 路由）拿到可选模型列表，用于页面下拉选择。
- 真正执行检测的是 `src/main.py` 里的 YOLO 模型加载逻辑。
- `auto` 会优先尝试本地模型（如 `models/best.pt`），没有就回退到 `yolo11n.pt`（由 Ultralytics 自动下载）。

## 为什么 GitHub 传不上去
- `*.pt/*.pth` 模型文件通常很大，超过 GitHub 普通仓库大小限制。
- `__pycache__`、`data/auth.db` 这类本地运行产物也不该进仓库。

## 推荐共享方式
- 代码仓库只放源码，不放大模型本体。
- 首次运行让程序自动下载内置模型（`yolo11n.pt`）。
- 若必须用自定义模型，把模型放网盘/Release，运行后手动放到 `models/`。

## 快速启动
```bash
python src/start.py
```