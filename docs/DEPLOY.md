# OCR 嵌入式部署指南

## 项目文件结构

```
cv_pc/                                 ← 项目根目录
│
├── ocr_workflow_accelerated.py        ★ PC端主程序（对外接口, 对标 asr.py）
│                                      └─ 6个接口: init_ocr / recognize_image / recognize_image_full
│                                           batch_recognize / set_ocr_callback / get_ocr_status
│
├── asr.py                             ASR语音识别（接口风格参考, 回调模式一致）
├── api_infer.py                       API推理入口
│
├── OCR接口.md                          ★ OCR对外接口使用文档（给队友看这个）
├── DEPLOY.md                          本文档（部署指南）
├── PROJECT_SUMMARY.md                 项目完整总结
│
├── ocr_workflow_onnx_pc.py            PC版OCR（旧版, 已被 accelerated 替代）
├── ocr_workflow_onnx_linux.py         Linux/开发板版OCR（业务逻辑来源）
├── ocr_board_v3.py                    板端加速版V3（推理引擎来源）
├── ocr_board.py                       板端版（更早版本）
│
├── camera_test.py / camera_test_improve.py / camera_test copy.py   摄像头相关
├── paddleocr_test.py                  PaddleOCR原版测试
├── test_easyocr_w8a8.py               EasyOCR W8A8量化测试
│
├── pp-ocrv4_rapid_onnx/               ★ ONNX模型目录（必须带！）
│   ├── ch_PP-OCRv4_det_mobile.onnx     ★ 文本检测 Det (~4.5MB)
│   ├── ch_PP-OCRv4_rec_mobile.onnx     ★ 文字识别 Rec 主用 (~10.3MB)
│   ├── ch_ppocr_mobile_v2.0_cls_mobile.onnx  ★ 方向分类 Cls (~0.5MB)
│   ├── ppocr_keys_v1.txt               ★ 中文字典 (6623字符)
│   │
│   ├── ch_PP-OCRv4_rec_mobile_int8.onnx    官方INT8量化 (实验性, 不用于生产)
│   ├── ch_PP-OCRv4_rec_mobile_custom_int8.onnx  自量化INT8 (实验性)
│   ├── *.dlc                            骁龙NPU格式 (开发板用, PC不用管)
│   └── infer/                           旧版infer模型 (未使用)
│
├── photos/                             测试图片 (18张药品说明书)
│
├── 板端数据（测试）/                   开发板测试数据与脚本
├── easyocr-onnx-w8a8/                 EasyOCR W8A8量化模型
├── easyocr_md/                        EasyOCR相关文档
├── yolo人体检测/                      YOLO人体检测模块
│
├── _leak_analysis.txt                 内存泄漏分析
├── 板端精度下降诊断报告.md             精度问题诊断
└── ocr_test_result.txt                OCR测试结果记录
```

### 核心文件说明

| 文件 | 角色 | 说明 |
|------|:----:|------|
| `ocr_workflow_accelerated.py` | **★ 主程序** | PC端最终整合版, 含6个对外接口 |
| `asr.py` | 接口参考 | ASR回调模式, OCR接口与其对齐 |
| `OCR接口.md` | **★ 使用文档** | 队友看这个就能调用OCR |
| `pp-ocrv4_rapid_onnx/` | **★ 模型** | 最少只需4个文件 (见下方) |

### PC部署最少携带

```
pp-ocrv4_rapid_onnx/
  ch_PP-OCRv4_det_mobile.onnx              (4.5 MB)
  ch_PP-OCRv4_rec_mobile.onnx              (10.3 MB)
  ch_ppocr_mobile_v2.0_cls_mobile.onnx     (0.5 MB)
  ppocr_keys_v1.txt                         (0.03 MB)
                                          ─────────
                                          总计 ~15.3 MB
```

---

## 各模型文件作用

### 必须携带的 4 个文件

| 文件名 | 大小 | 作用 | 在流程中的位置 |
|--------|:----:|------|:-------------:|
| `ch_PP-OCRv4_det_mobile.onnx` | 4.53 MB | **文本检测 (Det)** — 在图片中找到所有文字的位置框 | 第1步 |
| `ch_PP-OCRv4_rec_mobile.onnx` | 10.35 MB | **文字识别 (Rec)** — 对每个位置框内的图片识别出具体文字 | 第3步 |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | 0.56 MB | **方向分类 (Cls)** — 判断文字是否旋转了90°/180°，自动纠正 | 第2步 |
| `ppocr_keys_v1.txt` | 0.03 MB | **中文字典** — 6623个常用汉字，识别时逐字匹配 | 配合Rec使用 |

### 可删除的 4 个文件

| 文件名 | 为什么不需要 |
|--------|------------|
| `ch_PP-OCRv4_det_infer.onnx` | server/infer版检测模型，比mobile大且慢，config默认不加载 |
| `ch_PP-OCRv4_rec_infer.onnx` | server/infer版识别模型，同上 |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | infer版分类模型，同上 |
| `ppocrv5_dict.txt` | PP-OCRv5 的字典，当前项目使用的是 v4 |

> **最小部署包** 只需上述 4 个文件 = **15.47 MB**

---

## OCR 工作流程

```
输入图片 (BGR numpy array 或 图片路径)
        │
        ▼
  ┌─────────────┐
  │ ① Det 检测  │  ch_PP-OCRv4_det_mobile.onnx
  │  找文字位置  │  输出：N个四边形坐标框 [x1,y1,x2,y2,x3,y3,x4,y4]
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ ② Cls 分类  │  ch_ppocr_mobile_v2.0_cls_mobile.onnx
  │  判断旋转角度 │  输出：0° 或 180°（自动翻转）
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ ③ Rec 识别  │  ch_PP-OCRv4_rec_mobile.onnx + ppocr_keys_v1.txt
  │  读出文字内容 │  输出：文字字符串 + 置信度 (0~1)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ ④ 排序输出  │  纯代码逻辑（无模型）
  │  单列/双栏   │  根据坐标重新排列阅读顺序
  └─────────────┘
         │
         ▼
  返回: { texts[], scores[], boxes[], count, avg_score }
```

---

## 部署到新环境需要的依赖

### Python 环境

```bash
# 推荐 Python 3.8+ (本项目测试于 3.8)
python --version

# 安装依赖（无深度学习框架，纯 ONNX 推理）
pip install rapidocr onnxruntime opencv-python numpy psutil
```

### 依赖包大小参考

| 包名 | 大约大小 | 用途 |
|------|:-------:|------|
| `rapidocr` | ~2 MB | OCR 引擎封装 |
| `onnxruntime` | ~50-150 MB | ONNX 模型运行时 |
| `opencv-python` | ~40 MB | 图像处理 / 摄像头 |
| `numpy` | ~30 MB | 数值计算 |
| `psutil` | ~1 MB | 系统信息（可选） |

### 完整复制命令

把以下文件/文件夹复制到目标机器即可离线运行：

```bash
# 方法1：只拷贝必要文件（最省空间，~16MB 模型 + 代码）
cp ocr_workflow_onnx.py 目标目录/
cp -r pp-ocrv4_rapid_onnx/ 目标目录/
# 然后：pip install rapidocr onnxruntime opencv-python numpy

# 方法2：完整拷贝（包含测试图片等）
cp -r cv/ 目标目录/
# 然后：pip install rapidocr onnxruntime opencv-python numpy psutil
```

---

## 使用方式

```bash
# 默认模式：自动识别 photos/ 下所有图片
python ocr_workflow_onnx.py

# 指定图片识别
python ocr_workflow_onnx.py photo.jpg "说明书.jpg"

# 启用图像预处理（模糊/光照不均时推荐）
python ocr_workflow_onnx.py --preprocess photo.jpg

# 摄像头模式：实时预览 → 空格拍照 → 自动识别
python ocr_workflow_onnx.py --camera
```

---

## 当前已优化的参数

```python
OCR_PARAMS = {
    "Det.thresh": 0.20,        # 默认0.3 → 降低，更容易检出浅色文字
    "Det.box_thresh": 0.35,     # 默认0.5 → 降低，保留更多低置信度框
    "Global.text_score": 0.4,   # 默认0.5 → 降低最终输出阈值
}
```
这些参数针对**药品说明书**场景优化，减少小字体/浅色文字的漏检。

---

## mobile vs server 版本对比

| | mobile (当前使用) | server (infer) |
|--|:-:|:-:|
| 模型大小 | 小 (~5MB) | 大 (~15MB) |
| 推理速度 | **快** | 慢 |
| 精度 | 略低 | 略高 |
| 适用场景 | **嵌入式 / 边端设备** | 服务器 / 高性能PC |

嵌入式设备建议始终使用 **mobile** 版本。

---

## INT8 量化模型说明 (2026-04-27)

### 目录下有两个 INT8 文件, 不要混淆:

| 文件 | 大小 | 来源 | 校准数据 |
|:-----|:----:|:-----|:---------|
| `ch_PP-OCRv4_rec_mobile_int8.onnx` | **10.63 MB** | **PaddleOCR 官方模型库下载** | 通用数据集(非本项目图片) |
| `ch_PP-OCRv4_rec_mobile_custom_int8.onnx` | **10.6 MB** | **项目自量化 (quantize_rec_int8.py)** | photos/ 目录下的3张药品说明书 |

### 关键区别

| 维度 | 官方 INT8 (`_int8.onnx`) | 自定义 INT8 (`_custom_int8.onnx`) |
|:----:|:------------------------:|:--------------------------------:|
| **量化方式** | 可能是真正的 INT8 整数运算 | ORT QDQ 静态量化 (Quantize→FP32→Dequantize) |
| **PC端精度** | 未单独测试 | 余弦相似度 0.9655 (⚠️校准集仅3张, 偏差) |
| **PC端速度** | 未单独测试 | **慢 4.7x** (40.7ms vs FP32 8.6ms) |
| **板端精度** | **❌ 崩坏**: 58%空串+剩余乱码 | **未测试** |
| **板端速度** | 快 47% (7.7s vs FP32 11.4s) | **未测试** |

### 为什么官方 INT8 在板端崩坏了?

```
根因: 校准集不匹配
  PaddlePaddle 用通用数据集(文档/街景/标志等)做 INT8 校准
  → 生成的量化参数(scale/zero_point)适合通用场景
  → 但本项目的输入是"药品说明书"(特定字体、特定布局、特定颜色)
  → CTC 层输出的概率分布被破坏
  → 解码器输出大量 blank token → 58%文本变空串, 剩余乱码

类比: 用"英语考试"的评分标准去批改"中文作文"
```

### 为什么 QDQ 自量化在 PC 上反而更慢?

```
原因: CPU Execution Provider 不支持 QDQ 加速指令
  QDQ 格式 = QuantizeLinear → [原始FP32算子] → DequantizeLinear
  推理时流程:
    输入 → Quant(INT8→INT8) → 反量化(INT8→FP32) → FP32计算 → 量(FP32→INT8) → DQ(INT8→FP32) → 输出
    比纯FP32多了4次类型转换操作
  
  QDQ 本是为 GPU/NPU 设计的, 这些硬件有原生 INT8 计算单元
  x86/ARM 的普通 CPU 没有 → 只能模拟 → 更慢
```

### 当前推荐

```
生产环境继续用 FP32: ch_PP-OCRv4_rec_mobile.onnx (10.35MB)
  ✅ 精度可靠, PC和板端一致
  ✅ 速度基线明确 (PC ~8ms/张, 板端 ~124ms/张)

INT8 模型仅作实验参考, 不用于生产。
如需真正加速, 应走 P0/P1/P3 配置优化方向 (见 PROJECT_SUMMARY.md §11.5)
```
