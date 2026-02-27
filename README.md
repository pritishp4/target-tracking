# MTTrack 多目标智能追踪系统

[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/yourusername/mttrack)
[![Python](https://img.shields.io/badge/python-3.8+-green)](https://www.python.org/)

多目标实时追踪系统，支持结合视觉语言模型进行目标分类。

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [命令行使用](#命令行使用)
- [Python API 使用](#python-api-使用)
- [VL 模型配置](#vl-模型配置)
- [配置参数详解](#配置参数详解)
- [创新性说明](#创新性说明)
- [项目结构](#项目结构)
- [架构设计](#架构设计)
- [测试运行](#测试运行)
- [常见问题](#常见问题)

---

## 功能特性

### 基础功能
- **多种追踪算法**: 支持 ByteTrack 和 SORT 经典算法
- **YOLO 集成**: 支持任意 YOLO 模型 (YOLOv8, YOLOv10, YOLOv11 等)
- **VL 分类增强**: 可选的视觉语言模型，用于细粒度目标分类
- **模块化设计**: 清晰的领域层、服务层、基础设施层分离
- **视频读写**: 便捷的视频读写工具

### 增强模式 (v0.2.0+)
- **外观特征提取**: 轻量级外观 embedding，实现鲁棒的目标重识别
- **自适应 VL 触发**: 智能决策何时调用 VL 模型
- **多特征融合关联**: IoU + 外观 + 运动多信号联合决策
- **自适应阈值**: 根据场景密度动态调整关联阈值

---

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/mttrack.git
cd mttrack

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备 YOLO 模型

将 YOLO 模型文件放入 `yolo/` 目录，或指定完整路径：

```bash
# 示例：yolo/yolo26x.pt
```

### 3. 运行追踪

```bash
# 标准模式（最简单的使用方式）
python mttrack.py --input data/test.mp4 --output result.mp4
```

---

## 命令行使用

### 基础命令

```bash
# 标准模式 - 使用默认配置
python mttrack.py --input 输入视频路径 --output 输出视频路径

# 示例
python mttrack.py --input data/test_multi_target_tracker_video.mp4 --output out/result.mp4
```

### 增强模式

增强模式包含外观特征提取、自适应 VL 触发、多特征融合关联等功能：

```bash
python mttrack.py \
    --input data/test_multi_target_tracker_video.mp4 \
    --output out/result_enhanced.mp4 \
    --enhanced
```

### 完整参数示例

```bash
# 完整参数示例
python mttrack.py \
    --input data/test.mp4 \
    --output out/result.mp4 \
    --tracker bytetrack \
    --yolo-model ./yolo/yolo26x.pt \
    --confidence 0.25 \
    --device cuda \
    --enable-vl \
    --vl-interval 30 \
    --vl-timeout 30 \
    --enhanced \
    --no-appearance \
    --vl-min-interval 30 \
    --vl-max-interval 150
```

### 命令行参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入视频路径（必需） | - |
| `--output` | `-o` | 输出视频路径（必需） | - |
| `--tracker` | - | 追踪算法: `bytetrack` 或 `sort` | `bytetrack` |
| `--yolo-model` | - | YOLO 模型路径 | `./yolo/yolo26x.pt` |
| `--confidence` | - | 检测置信度阈值 | `0.25` |
| `--device` | - | YOLO 设备: `cuda`, `cpu`, `0`, `1` | `cuda` |
| `--enable-vl` | - | 启用 VL 分类（需要 VLLM 服务） | False |
| `--vl-interval` | - | VL 分类间隔帧数（标准模式） | `30` |
| `--vl-timeout` | - | VL API 超时秒数 | `30` |
| `--enhanced` | - | 启用增强模式 | False |
| `--no-appearance` | - | 增强模式下禁用外观特征 | False |
| `--vl-min-interval` | - | 增强模式 VL 最小间隔 | `30` |
| `--vl-max-interval` | - | 增强模式 VL 最大间隔 | `150` |

---

## Python API 使用

### 方式一：标准模式（推荐入门）

```python
from mttrack import (
    TrackerService,
    YoloDetector,
    TrackingAnnotator,
    VideoReader,
    VideoWriter
)

# 1. 初始化检测器
detector = YoloDetector(
    model_path="./yolo/yolo26x.pt",
    confidence_threshold=0.25,
    device="cuda"
)

# 2. 初始化追踪服务
tracker_service = TrackerService(
    detector=detector,
    tracker_type="bytetrack"  # 或 "sort"
)

# 3. 初始化标注器
annotator = TrackingAnnotator()

# 4. 处理视频
input_video = "input.mp4"
output_video = "output.mp4"

with VideoReader(input_video) as reader:
    with VideoWriter(output_video, fps=reader.fps) as writer:
        for frame_id, frame in reader:
            # 处理每一帧
            result = tracker_service.process_frame(frame)

            # 标注跟踪结果
            annotated = annotator.annotate(frame, result.tracks)

            # 写入输出视频
            writer.write(annotated)

print("处理完成！")
```

### 方式二：增强模式（推荐生产使用）

```python
from mttrack import (
    EnhancedTrackerService,  # 增强版追踪服务
    YoloDetector,
    TrackingAnnotator,
    VideoReader,
    VideoWriter
)
from mttrack.infrastructure import VllmClient

# 1. 初始化检测器
detector = YoloDetector(
    model_path="./yolo/yolo26x.pt",
    confidence_threshold=0.25,
    device="cuda"
)

# 2. 初始化 VL 客户端（可选，需要 VLLM 服务）
vllm_client = None
# vllm_client = VllmClient(
#     base_url="http://10.132.19.82:50100",
#     api_key="sk-xxx",
#     model="/models/Qwen/Qwen3-VL-8B-Instruct"
# )

# 3. 初始化增强版追踪服务
tracker_service = EnhancedTrackerService(
    detector=detector,
    tracker_type="bytetrack",
    enable_appearance=True,       # 启用外观特征提取
    enable_adaptive_vl=True,      # 启用自适应 VL 触发
    vl_min_interval=30,           # VL 最小间隔帧数
    vl_max_interval=150,          # VL 最大间隔帧数
)

# 4. 初始化标注器
annotator = TrackingAnnotator()

# 5. 处理视频
with VideoReader("input.mp4") as reader:
    with VideoWriter("output.mp4", fps=reader.fps) as writer:
        for frame_id, frame in reader:
            result = tracker_service.process_frame(frame)
            annotated = annotator.annotate(frame, result.tracks)
            writer.write(annotated)
```

### 方式三：结合 VL 分类（完整版）

```python
from mttrack import (
    EnhancedTrackerService,
    YoloDetector,
    TrackingAnnotator,
    VideoReader,
    VideoWriter
)
from mttrack.infrastructure import VllmClient
from mttrack.service import LabelService

# 1. 初始化 VL 客户端
vllm_client = VllmClient(
    base_url="http://your-vllm-server:50100",  # 替换为你的 VLLM 地址
    api_key="sk-your-api-key",
    model="/models/Qwen/Qwen3-VL-8B-Instruct"
)

# 2. 初始化检测器
detector = YoloDetector(
    model_path="./yolo/yolo26x.pt",
    confidence_threshold=0.25,
    device="cuda"
)

# 3. 初始化追踪服务
tracker_service = EnhancedTrackerService(
    detector=detector,
    tracker_type="bytetrack",
    enable_appearance=True,
    enable_adaptive_vl=True,
)

# 4. 初始化标签服务
label_service = LabelService(
    vllm_client=vllm_client,
    enabled=True,
    label_interval=30,
    cache_ttl=60.0
)

# 5. 处理视频
annotator = TrackingAnnotator()

with VideoReader("input.mp4") as reader:
    with VideoWriter("output.mp4", fps=reader.fps) as writer:
        for frame_id, frame in reader:
            result = tracker_service.process_frame(frame)

            # 获取活跃跟踪 ID
            active_track_ids = {t.track_id for t in result.tracks}

            # 清理过期缓存
            label_service.cleanup_old_tracks(active_track_ids)

            # VL 分类（如果启用）
            if vllm_client:
                for track in result.tracks:
                    # 检查是否需要分类
                    should_label, reason = tracker_service.should_classify_vl(
                        track.track_id,
                        track.bbox,
                        track.label_confidence if track.label_confidence > 0 else track.confidence
                    )

                    if should_label:
                        # 裁剪目标区域
                        crop = crop_track(frame, track.bbox)
                        if crop is not None and crop.size > 0:
                            # 调用 VL 分类
                            vl_result = label_service.label_track(
                                track.track_id,
                                crop,
                                frame_id
                            )
                            if vl_result and vl_result.class_name != "unknown":
                                # 更新标签
                                track.label = vl_result.class_name
                                track.label_confidence = vl_result.confidence
                                tracker_service.update_track_label(
                                    track.track_id,
                                    vl_result.class_name,
                                    vl_result.confidence
                                )

            # 标注并输出
            annotated = annotator.annotate(frame, result.tracks)
            writer.write(annotated)


def crop_track(frame, bbox, margin=10):
    """裁剪跟踪区域"""
    import numpy as np
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
```

### 高级：自定义检测器

```python
from mttrack.infrastructure import BaseDetector, DetectorResult
import numpy as np

class MyDetector(BaseDetector):
    """自定义检测器示例"""

    def __init__(self, model_path):
        # 这里加载你的模型
        pass

    def detect(self, image):
        # 返回检测结果
        return DetectorResult(
            boxes=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["person"]
        )

    def warmup(self):
        pass

# 使用自定义检测器
detector = MyDetector("my_model.pt")
tracker_service = TrackerService(detector=detector)
```

---

## VL 模型配置

### 什么是 VL 分类？

VL（Vision-Language）分类使用视觉语言大模型（如 Qwen2-VL）对跟踪目标进行细粒度分类，可以识别比 YOLO 更丰富的目标类别。

### 配置步骤

#### 1. 设置环境变量

```bash
# 方式一：命令行设置（临时）
export VLLM_BASE_URL="http://10.132.19.82:50100"
export VLLM_API_KEY="sk-your-api-key"
export VLLM_MODEL="/models/Qwen/Qwen3-VL-8B-Instruct"

# 方式二：一行命令（临时）
VLLM_BASE_URL="http://10.132.19.82:50100" \
VLLM_API_KEY="sk-your-api-key" \
VLLM_MODEL="/models/Qwen/Qwen3-VL-8B-Instruct" \
python mttrack.py --input video.mp4 --output result.mp4 --enable-vl
```

#### 2. 环境变量说明

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `VLLM_BASE_URL` | VLLM API 服务器地址 | `http://10.132.19.82:50100` |
| `VLLM_API_KEY` | API 密钥（可随意设置） | `sk-xxxx` |
| `VLLM_MODEL` | VL 模型名称 | `/models/Qwen/Qwen3-VL-8B-Instruct` |

#### 3. 验证 VL 是否启用

运行后观察日志：
- **启用成功**：
  ```
  [Info] Initializing VLLM client...
  [Info] VLLM client initialized (base_url: http://xxx:50100)
  ```
- **启用失败**：
  ```
  [Warning] VLLM_BASE_URL not set, VL classification disabled
  ```

---

## 配置参数详解

### YOLO 检测器参数

```python
detector = YoloDetector(
    model_path="./yolo/yolo26x.pt",  # 模型路径
    confidence_threshold=0.25,        # 置信度阈值（0-1）
    device="cuda"                    # 设备：cuda/cpu/0/1
)
```

### ByteTrack 参数

```python
tracker = ByteTrackTracker(
    lost_track_buffer=30,           # 丢失缓冲帧数
    frame_rate=30.0,                 # 帧率
    track_activation_threshold=0.7, # 轨迹激活阈值
    minimum_consecutive_frames=2,   # 最小连续帧数
    minimum_iou_threshold=0.1,      # 最小 IoU 阈值
    high_conf_det_threshold=0.6      # 高置信度检测阈值
)
```

### SORT 参数

```python
tracker = SORTTracker(
    lost_track_buffer=30,
    frame_rate=30.0,
    track_activation_threshold=0.25,
    minimum_consecutive_frames=3,
    minimum_iou_threshold=0.3
)
```

### 增强模式参数

```python
tracker = EnhancedTrackerService(
    detector=detector,
    tracker_type="bytetrack",

    # 外观特征
    enable_appearance=True,
    appearance_memory_size=10,

    # 自适应 VL
    enable_adaptive_vl=True,
    vl_min_interval=30,
    vl_max_interval=150,

    # 多特征关联
    enable_multi_feature=True,
    use_appearance_in_association=True,
)
```

---

## 创新性说明

### 本系统核心创新点（可用于软件著作权申请）

#### 1. 自适应 VL 智能触发策略

**创新描述**：传统 VL 分类采用固定间隔触发，本系统提出基于多维信号融合的自适应触发策略。

**技术方案**：
- **时间信号**：距离上次分类的帧数
- **运动信号**：目标从静止到运动的变化、速度突变检测
- **外观信号**：基于颜色直方图的外观变化幅度计算
- **置信度信号**：YOLO 检测置信度和历史 VL 分类置信度

**效果**：显著减少不必要的 VL API 调用，提升实时性。

#### 2. 轻量级外观特征提取与重识别

**创新描述**：提出基于颜色直方图（HSV空间）+ 边缘梯度特征（HOG-like）的轻量级外观 embedding 方法。

**技术方案**：
- HSV 颜色空间 32 bin 直方图（6 个通道：BGR + HSV）
- Sobel 梯度方向直方图（HOG 简化版）
- 余弦相似度计算
- 历史特征滑动平均

**效果**：在目标遮挡后重识别场景，ID 跳变率显著降低。

#### 3. 多特征融合数据关联

**创新描述**：突破传统 IoU 单一特征关联，提出多信号联合决策。

**技术方案**：
- IoU 相似度（权重 40%）
- 外观特征相似度（权重 35%）
- 运动一致性（速度相似度，权重 15%）
- 尺寸相似度（权重 10%）

**效果**：复杂场景（遮挡、密集目标、快速运动）下的跟踪鲁棒性提升。

#### 4. 自适应关联阈值

**创新描述**：根据场景密度和运动状态动态调整 IoU 关联阈值。

**技术方案**：
- 目标密度高 → 降低阈值（更严格）
- 运动速度快 → 降低阈值（更严格）
- 使用指数平滑跟踪历史

**效果**：自动适应不同场景，避免误匹配。

---

## 项目结构

```
mttrack/
├── mttrack/                      # 主包
│   ├── __init__.py              # 包初始化
│   ├── domain/                   # 领域层（核心算法）
│   │   ├── __init__.py
│   │   ├── models.py            # 数据模型
│   │   ├── tracker.py           # 追踪器基类
│   │   ├── kalman.py           # 卡尔曼滤波器
│   │   ├── bytetrack.py        # ByteTrack 实现
│   │   ├── sort.py             # SORT 实现
│   │   ├── appearance.py       # 【创新】外观特征提取
│   │   ├── adaptive_trigger.py  # 【创新】自适应VL触发
│   │   └── association.py      # 【创新】多特征融合关联
│   │
│   ├── infrastructure/          # 基础设施层（外部集成）
│   │   ├── __init__.py
│   │   ├── detector.py         # YOLO 检测器
│   │   ├── vllm_client.py      # VLLM 客户端
│   │   └── video_io.py         # 视频读写
│   │
│   ├── service/                  # 服务层（业务逻辑）
│   │   ├── __init__.py
│   │   ├── tracker_service.py  # 标准追踪服务
│   │   ├── label_service.py    # VL 标签服务
│   │   └── enhanced_tracker_service.py  # 【创新】增强版追踪服务
│   │
│   └── annotators/               # 可视化层
│       ├── __init__.py
│       └── tracking_annotator.py
│
├── tests/                        # 单元测试
│   ├── test_domain.py
│   ├── test_infrastructure.py
│   └── test_service.py
│
├── mttrack.py                    # 命令行入口
├── README.md                     # 本文档
└── requirements.txt              # 依赖
```

---

## 架构设计

项目采用经典的**分层架构**：

```
┌─────────────────────────────────────────────────────────┐
│                    接口层 (Interface)                   │
│                   mttrack.py / CLI                      │
├─────────────────────────────────────────────────────────┤
│                    服务层 (Service)                      │
│     TrackerService  │  LabelService  │  Enhanced        │
├─────────────────────────────────────────────────────────┤
│                  领域层 (Domain)                         │
│   ByteTrack │ SORT │ Kalman │ Appearance │ Adaptive     │
├─────────────────────────────────────────────────────────┤
│               基础设施层 (Infrastructure)                │
│       YOLO Detector │ VLLM Client │ Video I/O          │
└─────────────────────────────────────────────────────────┘
```

### 各层职责

- **领域层**：核心追踪算法和数据模型，包含原创的创新模块
- **基础设施层**：外部系统集成（YOLO、VLLM、文件系统）
- **服务层**：业务逻辑编排，协调各模块工作
- **接口层**：命令行入口和可视化输出

---

## 测试运行

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_domain.py

# 带覆盖率
pytest --cov=mttrack --cov-report=html
```

---

## 常见问题

### Q1: VL 模型没有启用？

**A**: 检查环境变量是否设置：
```bash
echo $VLLM_BASE_URL  # 应该显示地址
```

### Q2: GPU 不可用？

**A**: 使用 CPU 模式：
```bash
python mttrack.py --input video.mp4 --output result.mp4 --device cpu
```

### Q3: 追踪效果不好？

**A**: 尝试增强模式：
```bash
python mttrack.py --input video.mp4 --output result.mp4 --enhanced
```

### Q4: 视频太大处理太慢？

**A**:
1. 使用 GPU：`--device cuda`
2. 降低分辨率：使用 FFmpeg 预处理
3. 跳过帧：修改代码中的迭代逻辑

---

## 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE)

---

## 致谢

- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 多目标追踪
- [SORT](https://github.com/abewley/sort) - 简单在线实时追踪
- [Supervision](https://github.com/roboflow/supervision) - 计算机视觉工具
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 框架

---

## 联系方式

如有问题，请提交 Issue 或联系作者。
