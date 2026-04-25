# OCR 嵌入式部署指南

## 项目文件结构

```
cv/                                    ← 项目根目录
│
├── ocr_workflow_onnx.py               ★ 主程序（拍照+预处理+OCR）
├── camera_test.py                      摄像头独立测试（备用）
├── paddleocr_test.py                   PaddleOCR 测试（备用，需联网下载模型）
│
├── pp-ocrv4_rapid_onnx/                ★ ONNX 模型目录（必须带！）
│   ├── ch_PP-OCRv4_det_mobile.onnx     ★ 文本检测模型
│   ├── ch_PP-OCRv4_rec_mobile.onnx     ★ 文字识别模型
│   ├── ch_ppocr_mobile_v2.0_cls_mobile.onnx  ★ 方向分类模型
│   ├── ppocr_keys_v1.txt               ★ 中文字典（6623个字符）
│   │
│   ├── ch_PP-OCRv4_det_infer.onnx      ✗ 不需要（server版，未使用）
│   ├── ch_PP-OCRv4_rec_infer.onnx      ✗ 不需要（server版，未使用）
│   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx  ✗ 不需要（infer版，未使用）
│   └── ppocrv5_dict.txt                ✗ 不需要（v5字典，当前用的是v4）
│
└── photos/                             测试图片目录
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
