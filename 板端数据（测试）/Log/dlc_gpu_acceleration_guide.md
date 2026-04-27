# Qualcomm 开发板 DLC/GPU/NPU 硬件加速经验文档

> 项目: OCR Board (PP-OCRv4 Rapid)
> 平台: QCS6490-ODK (Snapdragon, Adreno GPU)
> SDK: fiboaisdk (SNPE/QNN 封装)
> 更新时间: 2026-04-26

---

## 目录

1. [硬件环境与SDK接口](#1-硬件环境与sdk接口)
2. [DLC 调度能力验证](#2-dlc-调度能力验证)
3. [各阶段GPU实测结果](#3-各阶段gpu实测结果)
4. [Det-GPU 失败的根因分析（核心难点）](#4-detgpu-失败的根因分析核心难点)
5. [Cls-GPU 精度问题](#5-clsgpu-精度问题)
6. [ORT-CPU 是什么](#6-ortcpu-是什么)
7. [性能瓶颈分析与加速方案](#7-性能瓶颈分析与加速方案)
8. [最终选型决策与原因](#8-最终选型决策与原因)
9. [后续可探索的方向](#9-后续可探索的方向)
10. [调试工具与方法论](#10-调试工具与方法论)

---

## 1. 硬件环境与SDK接口

### 1.1 硬件平台
```
芯片: QCS6490 (Snapdragon)
GPU: Adreno 600 系列 (支持 OpenCL / SNPE / QNN)
NPU/DSP: Hexagon (需额外推送库文件)
CPU: ARM Cortex (多核)
```

### 1.2 SDK 接口 (`api_infer.py`)

```python
from fiboaisdk.api_aisdk_py import api_infer_py

session = InferenceSession(
    model="/path/to/model.dlc",       # DLC模型路径(必须绝对路径)
    platform="qualcomm",               # 固定值
    framework="snpe",                  # 或 "qnn"
    runtime="CPU",                     # 可选: "CPU" | "GPU" | "DSP" | "NPU"
    log_level="ERROR",                 # 日志级别
    profile_level=5,                   # 性能档位 (5=BURST, 高性能模式)
)

ret = session.Initialize()             # 返回 0 = 成功
result = session.Execute(
    output_names=["output_name"],      # 需要的输出张量名列表
    input_feed={"input_name": data},   # 输入数据字典
)
session.Destroy()
```

### 1.3 关键注意事项
- **model 路径必须是绝对路径**，相对路径会导致 Initialize 失败
- **DSP 运行时需要预先推送 Hexagon 库文件到工作目录**
- **profile_level 含义**: SNPE用0~9(5=BURST), QNN用0~13
- **GPU 初始化会有警告日志**（见下方），属于正常现象

---

## 2. DLC 调度能力验证

### 2.1 GPU 初始化日志（正常）

每次调用 `runtime="GPU"` 时，系统日志会输出以下信息：
```
GPU ERROR: GPU_ERROR_UNSUPPORTED(10018) - Setting context priority after context initialization not supported
QnnContext_setConfig() failed; QNN_CONTEXT_ERROR_INVALID_ARGUMENT
```
**这不是致命错误！** 模型仍然可以正常加载和推理。这些是 QNN 后端在设置 GPU 上下文优先级时的非关键警告。

### 2.2 已验证可运行的配置

| 模型 | 运行时 | 状态 | 推理速度(单次) |
|------|--------|------|---------------|
| Det (PP-OCRv4_det_mobile) | CPU | 可运行，输出有损 | ~0.3s |
| Det (PP-OCRv4_det_mobile) | GPU | **可运行但输出噪声** | ~0.1s |
| Cls (mobile_v2.0_cls) | GPU | **可运行但有误分类** | ~0.05-0.10s |
| Cls (mobile_v2.0_cls) | CPU | 可运行 | ~0.15s |
| Rec (MobileOne) | 无法转DLC | N/A | N/A |

### 2.3 DLC模型转换来源
- ONNX → DLC 的转换使用高通官方工具链 (snpe-onnx-to-dlc 或类似工具)
- 当前 DLC 模型疑似为 **INT8 量化版本**（未确认转换参数）
- 模型存放在: `pp-ocrv4_rapid_onnx/*.dlc`

---

## 3. 各阶段 GPU 实测结果

### 3.1 Det 阶段对比

| 配置 | 输出框数 | Avg Score | 推理时间 | 质量 |
|------|---------|-----------|---------|------|
| ORT-CPU (基准) | **88** | 0.95+ | 1.55-1.70s | 完美 |
| DLC-CPU 单次输入 | 5-15 | 0.30-0.50 | ~0.3s | 大量漏检 |
| DLC-GPU 单次输入 | 0-8 | <0.20 | ~0.1s | **全噪声** |
| DLC-CPU 多图块(最优参数) | 88-89 | 0.65* | ~28s | 框数量OK但Rec差 |

*DLC多图块的 score 是检测框置信度，不是Rec识别分数。实际Rec成功率仅~38%。

### 3.2 Cls 阶段对比

| 配置 | 误旋转数量 | 误旋转率 | 推理时间 | 质量 |
|------|-----------|---------|---------|------|
| ORT-CPU | 0/88 | 0% | 0.31s | 完美 |
| DLC-GPU | 20/88 | 22.7% | 0.05-0.10s | 不可接受 |
| DLC-CPU | 未详细测试 | - | ~0.15s | 待验证 |

### 3.3 Rec 阶段
- MobileOne 架构无法转换为 DLC（算子不支持）
- 固定使用 ONNX Runtime + CPU
- 88 个框串行推理约 10.86s（占总时间 84%）

---

## 4. Det-GPU 失败的根因分析（核心难点）

这是本项目遇到的最重要技术难题。

### 4.1 根因：固定输出分辨率导致的信息损失

```
┌─────────────────────────────────────────────────────────────┐
│                    ORT-CPU Det 流程                          │
│                                                             │
│  原图 1080x1920                                             │
│       ↓ resize (保持比例)                                    │
│  输入 736x1314 (limit_side_len=736, type='min')             │
│       ↓ DB 检测网络推理                                      │
│  输出 736x1314 概率图 (每个像素对应原图精确位置)              │
│       ↓ 二值化 → 轮廓提取 → 外接矩形                         │
│  结果: 88个精确文字框                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    DLC-GPU Det 流程                           │
│                                                             │
│  原图 1080x1920                                             │
│       ↓ 强制 resize                                         │
│  输入 640x640 (DLC模型固定输入/输出分辨率!)                  │
│       ↓ DB 检测网络推理                                      │
│  输出 640x640 概率图 (大量细节已被压缩丢失)                   │
│       ↓ 二值化 → 轮廓提取                                    │
│  结果: 仅检出大字体区域, 小字全部丢失 (<10框)                 │
└─────────────────────────────────────────────────────────────┘
```

**核心矛盾**: DLC 模型的计算图固化了 640×640 的输出尺寸，无法像 ORT 那样动态调整。
原图被压缩到 640×640 时，像素信息损失约 **(1080×1920)/(640×640) ≈ 5倍**，小字号文字的概率响应完全消失。

### 4.2 尝试过的解决方案及结果

#### 方案A：降低阈值补偿
- 方法：将 DET_THRESH 从 0.20 降到 0.10, BOX_THRESH 降到 0.20
- 结果：框数量略有增加但仍远低于 88，且引入大量假阳性
- 结论：**无效** — 信息已经丢失，阈值无法恢复

#### 方案B：概率图上采样 (Upscale)
- 方法：将 640×640 的概率图插值放大回原始分辨率再做后处理
- 测试了 INTER_LINEAR 和 INTER_CUBIC 两种插值
- 结果：上采样不能恢复已丢失的高频细节
- 结论：**无效**

#### 方案C：直接 640×640 输入（无预处理缩放）
- 方法：跳过 ORT 的智能 resize，直接把图压到 640×640
- 结果：与方案A类似，信息压缩太严重
- 结论：**无效**

#### 方案D：多图块检测 (Multi-tile) ⭐ 主要尝试
- 方法：将大图切成多个 640×640 重叠小块，分别检测后合并

```python
# 伪代码示意
tile_size = 640
overlap = 0.20  # 20%重叠
for each tile in split_image(img, tile_size, overlap):
    prob_map = dlc_detect(tile)        # 每个 tile 获得完整分辨率
    boxes = postprocess(prob_map)       # tile 内坐标
    boxes = map_to_global(boxes)        # 映射回原图坐标
all_boxes = nms_merge(all_tile_boxes)   # NMS 去重
```

- **调优过程**（详见 `_debug_fast.py` → `_debug_fast_v3.py`）:
  - 测试了 overlap ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
  - 测试了 thresh ∈ {0.12, 0.15, 0.20, 0.25}
  - 测试了 box_thresh ∈ {0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55}
  - 测试了 score_filter ∈ {None, 0.30~0.70}

- **最优配置**:
  ```
  TILE_SIZE=640, OVERLAP=0.20
  THRESH=0.25, BOX_THRESH=0.55, SCORE_FILTER=0.696
  → 输出 88 框 (匹配目标数量!)
  ```

- **但是!** Rec 阶段识别率暴跌:
  - 88 个框中仅 ~33 个能正确识别文字 (成功率 ~38%)
  - 其余 55 个框的 Rec 输出为空字符串或乱码
  - 平均 Rec 分数从 0.949 跌至 ~0.65

#### 多图块为什么最终失败？

经过 `_debug_box_detail.py`, `_debug_crop_diag.py`, `_debug_rec_raw.py`, `_debug_box_compare.py` 四轮诊断:

1. **框坐标偏差**: 多图块合并后的框坐标 vs ORT 基准有 1-5 像素的系统性偏移
2. **crop 质量下降**: 偏移导致的 crop 图像边缘截断或模糊（虽然人眼看着还行）
3. **CTC 解码敏感**: CTC-based 文字识别对 crop 质量高度敏感，轻微偏移就可能导致 blank token
4. **累积误差**: Tile 边界处的框拼接精度最低，恰好很多文字落在边界附近

**结论: 这是一个模型架构级别的限制，不是工程调参能解决的。**

### 4.3 解决此问题的前提条件

要让 Det 走 GPU 并达到 ORT 精度，需要满足以下任一条件:
1. **获取支持动态分辨率的 Det-DLC 模型** (转换时指定 dynamic shape)
2. **使用更高分辨率的 DLC 模型** (如 1280×1280，减少压缩比)
3. **更换检测算法**: 使用 YOLO-based 文本检测器 (天然固定分辨率友好)
4. **上游图像预处理**: 在送入 OCR 前，先将高分辨率图切分成局部区域分别 OCR（改变整体架构）

---

## 5. Cls-GPU 精度问题

### 5.1 问题现象

DLC-GPU Cls 对同一组 88 个文字 crop 的分类结果与 ORT-Cpu 存在显著差异：

| 统计项 | ORT-CPU | DLC-GPU |
|--------|---------|---------|
| 总判断数 | 88 | 88 |
| keep(不旋转) | 88 | 68 |
| ROT180(180°旋转) | 0 | 20 |
| 一致性基准 | 100% | 77.3% |

### 5.2 误旋转的影响链

```
DLC-Cls 错误判定 "ROT180"(置信度 0.906~0.998)
    ↓
crop 图像被 cv2.rotate(ROT180_180) 翻转
    ↓
正常文字变成倒置文字
    ↓
Rec 模型 (CTC解码) 对倒置文字输出 100% blank token
    ↓
score = 0.000 < TEXT_SCORE_THRESH(0.4)
    ↓
该框被过滤掉!
    ↓
88 框 → 最终仅 71 个文本输出 (损失 17 个)
```

### 5.3 可能的原因

1. **INT8 量化损失**: DLC 模型转换时使用了 INT8 量化，Cls 这种二分类任务对量化敏感
2. **GPU 浮点精度差异**: Adreno GPU 使用 FP16 计算，某些边界 case 的 softmax 输出可能翻转
3. **模型转换工具链 bug**: snpe-onnx-to-dlc 转换过程中可能存在算子等价性问题
4. **输入预处理细微差异**: DLC 和 ORT 的归一化/resize 实现可能有 sub-pixel 级别差异

### 5.4 缓解方案（未实施，待验证）

- **方案A**: 将 CLS_THRESH 从 0.90 提高到 0.999+，只在极高置信度时才信任 DLC 判断
- **方案B**: 双引擎校验 — 只有当 DLC 和 ORT 都判定旋转时才执行旋转
- **方案C**: 用 FP32 精度重新转换 Cls-DLC 模型 (`--float32` 参数)
- **方案D**: 直接禁用 DLC-Cls，Cls 本身只占 0.31s (总时间 2.4%)

---

## 6. ORT-CPU 是什么

### 6.1 定义

ORT-CPU = **ONNX Runtime + CPU 执行后端**

ONNX Runtime 是微软推出的跨平台推理引擎，支持:
- 多种操作系统 (Linux/Windows/Android/iOS)
- 多种硬件后端 (CPU/CUDA/CoreML/DirectML/OpenVINO/TensorRT)
- 算子融合、内存优化、并行执行等自动优化

### 6.2 我们的使用方式

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 4          # 内部并行线程数
so.inter_op_num_threads = 1           # 操作间并行
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(model_path.onnx, so)
outputs = session.run(None, {"input_name": input_data})
```

### 6.3 为什么 ORT-CPU 已经够快

| 优化手段 | 说明 |
|---------|------|
| 算子融合 | Conv+BN+ReLU 合并为单个节点，减少内存读写 |
| 内存复用 | In-place 操作减少分配开销 |
| 并行化 | intra_op_num_threads=4 利用多核CPU |
| 缓存友好 | 连续内存布局提升 cache 命中率 |

对于 PP-OCRv4 Rapid 这种 mobile 级轻量模型，ORT-CPU 在 ARM 多核上的效率已经很高。GPU 加速的优势主要体现在**大规模并行计算**（如大 batch、高分辨率），而 OCR 场景的特征是:
- 输入分辨率中等 (640-736)
- Batch size 小 (通常 1)
- 模型参数量小 (mobile 级)
→ GPU 并行优势不明显，反而通信开销可能成为瓶颈

---

## 7. 性能瓶颈分析与加速方案

### 7.1 当前耗时分布 (Total = 12.95s, ORT-CPU 全流程)

```
Rec:  ████████████████████████████████████████ 10.86s (84%) ← 绝对瓶颈!
Det:  ████ 1.70s (13%)
Cls:  ▌ 0.31s (2%)
其他: ▌ 0.08s (1%)
```

### 7.2 加速方案矩阵

| 优先级 | 方案 | 预期效果 | 实施难度 | 风险 |
|--------|------|---------|----------|------|
| **P0** | Rec INT8 量化模型 | 10.86s → 5-7s (-35%~50%) | 低(换模型即可) | 极低(ORT原生支持INT8) |
| **P1** | 增大 Rec Batch Size | 减少 Python 循环开销 | 低(改一行配置) | 低 |
| **P2** | 多线程流水线 (Det\|\|Rec) | 吞吐量翻倍(视频流场景) | 中(需重构pipeline) | 中 |
| **P3** | Cls 恢复 DLC-GPU (提高阈值) | 0.31s → 0.10s | 中(需实验找最佳阈值) | 中(可能仍有少量误判) |
| **P4** | FP32 重转换 Cls-DLC | 解决 Cls 精度根本问题 | 中(需重跑转换工具链) | 低 |
| **P5** | Rec 模型替换 (SVTR-lite) | 更轻量的识别骨架 | 高(需训练/找预训练模型) | 高(需验证精度) |
| **P6** | 动态分辨率 Det-DLC | 解决 Det GPU 硬伤 | 高(需上游提供新模型) | 低(如果拿到好模型) |

### 7.3 P0 方案详情: Rec INT8 量化

代码中已预留接口:
```python
# ocr_board.py 第81-82行
REC_ONNX = 'ch_PP-OCRv4_rec_mobile.onnx'     # 当前: FP32
# REC_ONNX = 'ch_PP-OCRv4_rec_mobile_int8.onnx'  # ← 切换到此行即可启用INT8
```

**实施步骤:**
1. 获取/转换 INT8 版本的 Rec ONNX 模型
2. 修改 `REC_ONNX` 配置指向新模型
3. ORT 自动以 INT8 模式运行（无需改代码逻辑）
4. 验证精度损失是否可接受（预期 < 1%）

**INT8 转换方法:**
```bash
# 使用 onnxruntime 的量化工具
python -m onnxruntime.quantization.dynamic_quantize_onnx_model \
    ch_PP-OCRv4_rec_mobile.onnx \
    ch_PP-OCRv4_rec_mobile_int8.onnx
```

---

## 8. 最终选型决策与原因

### 8.1 当前生产配置

```python
USE_DLC_DET = False     # Det: ORT-CPU (高精度 88 框)
USE_DLC_CLS = False     # Cls: ORT-CPU (零误旋转)
REC_ORT_RUNTIME = "CPU" # Rec: ORT-CPU (唯一选项)
```

### 8.2 决策依据

| 维度 | ORT-CPU (当前选择) | DLC-GPU/NPU |
|------|-------------------|-------------|
| **正确性** | 88/88 文本, score=0.949 | 71/88 文本 (Det) / 20/88误分类(Cls) |
| **速度** | 12.95s | 理论更快但数据不可用 |
| **稳定性** | 经过充分验证，结果可复现 | 依赖模型质量+运行时精度 |
| **维护成本** | 低，标准 ONNX 生态 | 中，需关注 DLC 兼容性 |

**核心理念: 正确性 > 速度。** 一个输出71个错误结果的快速系统，不如一个输出88个正确结果的慢系统。

### 8.3 这不是永久放弃 GPU

当前选择 CPU 方案是基于**现有模型和工具链约束**下的最优解。如果条件变化:
- ✅ 拿到动态分辨率 Det-DLC → 立即切换 GPU-Det
- ✅ FP32 Cls-DLC 验证通过 → 切换 GPU-Cls
- ✅ DSP/NPU 库部署完成 → 可尝试 DSP 运行时 (更低功耗)
- ✅ 新一代 SDK 支持 MobileOne 算子 → Rec 也能走 GPU

---

## 9. 后续可探索的方向

### 9.1 短期 (立即可做)
- [ ] P0: Rec INT8 量化（预计 -35%~50% Rec 时间）
- [ ] P1: 调优 REC_BATCH_NUM 减少调用开销
- [ ] 验证 `ch_PP-OCRv4_rec_mobile_int8.onnx` 是否可用

### 9.2 中期 (需实验)
- [ ] P3: Cls 提高阈值方案 (CLS_THRESH=0.99+) 的 A/B 测试
- [ ] P4: 用 FP32 重新转换 Cls-DLC 模型并测试
- [ ] DSP 运行时的可行性测试 (需先推送 Hexagon 库)

### 9.3 长期 (需外部依赖)
- [ ] 获取/制作动态分辨率版本的 Det-DLC 模型
- [ ] 评估 SVTR-lite 等 GPU-friendly 的 Rec 骨架
- [ ] 端侧 TFLite 方案 (如果 SDK 支持)
- [ ] 视频流场景的多帧并行 pipeline 架构

---

## 10. 调试工具与方法论

### 10.1 本次调试中创建的诊断脚本

所有脚本位于 `/home/fibo/cv/`，按调试阶段组织:

#### Phase 1: 基准建立
| 脚本 | 用途 |
|------|------|
| `_debug_dlc_system.py` | DLC vs ORT 全流程对比 (首次发现差距) |

#### Phase 2: Det 多图块方案
| 脚本 | 用途 |
|------|------|
| `_debug_optimize.py` | 初步参数搜索 (threshold grid search) |
| `_debug_optimize_v2.py` | NMS 改进 (矩形 IoU) |

#### Phase 3: Det 参数精细调优
| 脚本 | 用途 |
|------|------|
| `_debug_fast.py` | 扩展参数网格搜索框架 |
| `_debug_fast_v2.py` | 更广范围参数扫描 |
| `_debug_fast_v3.py` | 最优配置附近的精细微调 |
| `_debug_finetune.py` | 最优解邻域搜索 |
| `_debug_score_scan.py` | score_filter 精确值定位 |
| `_debug_precise.py` | 最终确认 0.696→88 框 |

#### Phase 4-5: 根因诊断 (最关键!)
| 脚本 | 用途 | 发现 |
|------|------|------|
| `_debug_gap.py` | 分析 DLC vs ORT 框位置差异 | 坐标系统性偏移 |
| `_debug_missing.py` | 分析漏检框特征 | 小字/密集区易丢失 |
| `_debug_det_params.py` | **证明 Det=88 框不是参数问题!** | 排除了参数假设 |
| `_debug_diagnose.py` | A/B 方法对比 | 确认两种方式Det输出一致 |
| `_debug_box_detail.py` | **逐框 Rec 分数分析** | 发现 16 个空字符串! |
| `_debug_crop_diag.py` | Crop 图片质量诊断 | 图片正常，排除裁剪问题 |
| `_debug_rec_raw.py` | **原始 Rec 模型输出** | 100% blank token! |
| `_debug_box_compare.py` | vs RapidOCR 坐标对比 | 坐标完全一致! |
| `_debug_cls_culprit.py` | **★ Cls 误旋转假设验证** | 16/16 全被 Cls 搞坏! |
| `_debug_cls_vs_ort.py` | **★ DLC vs ORT-Cls 对比** | 20 个 DLC 独有误旋转! |

### 10.2 诊断方法论总结

本次排查遵循的思路（可供未来参考）:

```
Step 1: 建立可信基准
  └─ ocr_workflow_onnx.py (RapidOCR) 作为 ground truth

Step 2: 定位问题发生的阶段
  └─ 逐层隔离: Det输出? → Cls输出? → Rec输出?
  └─ 工具: _debug_det_params.py, _debug_diagnose.py

Step 3: 数据级诊断 (不是猜测!)
  └─ 逐样本/逐框分析: 谁失败了? 失败的模式是什么?
  └─ 工具: _debug_box_detail.py, _debug_crop_diag.py

Step 4: 对比实验 (A/B Test)
  └─ 有Cls vs 无Clas → 定位 Cls 是元凶
  └─ DLC-Cls vs ORT-Cls → 确认是 DLC 模型本身的问题
  └─ 工具: _debug_cls_culprit.py, _debug_cls_vs_ort.py

Step 5: 根因确认后才做决策
  └─ 不是"感觉GPU不行"，而是"DLC模型X在Y条件下产生Z误差"
  └─ 有具体数字: 20/88误旋转, score=0.906-0.998
```

### 10.3 关键教训

1. **不要假设 GPU 一定更快更好** — 必须用数据验证每个阶段的输出正确性
2. **速度和精度的 tradeoff 要显式管理** — 先保证正确性，再优化速度
3. **DLC 模型的固定分辨率是一个常见陷阱** — 转换前要确认 dynamic shape 支持
4. **INT8 量化对分类/分割任务影响更大** — 对检测影响相对较小（我们观察到的是相反情况）
5. **多图块方案的累积误差容易被低估** — 每个tile的 sub-pixel 误差会在 NMS 合并时放大
6. **CTC-based Rec 对输入质量极其敏感** — 即使是人眼看不出来的 crop 偏移也可能导致 blank output

---

## 附录

### A. 文件索引

| 文件 | 说明 |
|------|------|
| `ocr_board.py` | 主程序 (1656行), 包含完整 OCR 引擎 |
| `ocr_workflow_onnx.py` | RapidOCR 基准实现 (529行) |
| `api_infer.py` | SDK 推理封装层 (255行) |
| `Log/dlc_det_optimization_log.txt` | 优化过程详细日志 |
| `Log/dlc_gpu_acceleration_guide.md` | ★ 本文档 ★ |
| `pp-ocrv4_rapid_onnx/` | 模型目录 (.onnx + .dlc) |
| `photos/photo_1.jpg` | 测试图片 (药品说明书) |
| `_debug_*.py` | 19个诊断脚本 (见上方表格) |

### B. 关键配置速查

```python
# 当前生产配置 (2026-04-26)
USE_DLC_DET = False
USE_DLC_CLS = False
DET_LIMIT_SIDE_LEN = 736
DET_LIMIT_TYPE = 'min'
DET_THRESH = 0.20
DET_BOX_THRESH = 0.35
CLS_THRESH = 0.90
TEXT_SCORE_THRESH = 0.4
intra_op_num_threads = 4

# 目标指标 (已达)
Texts: 88/88 (100%)
Avg Score: 0.9492
Total Time: 12.95s
```

### C. 参考资源

- SNPE Documentation: https://docs.qualcomm.com/blogs/snpe/
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/quantization.html
- PP-OCRv4 Paper: https://arxiv.org/abs/...
- 本项目 SDK 来源: fiboaisdk (api_infer_py)

---

## 11. Rec INT8 量化测试 (2026-04-26)

> 测试目的：验证 PaddleOCR 官方提供的 `ch_PP-OCRv4_rec_mobile_int8.onnx` 是否能在 ARM 上实现加速且保持精度。

### 11.1 测试环境

| 项目 | 值 |
|------|-----|
| CPU | Snapdragon, 8核 |
| ORT 版本 | 1.20.x |
| 线程数 | intra_op=4, inter_op=1 (已验证最优) |
| Batch size | 16 |
| 测试图 | photo_1.jpg → Det检出74框 |

### 11.2 对比数据（原始 CTC 输出，未做 score 阈值过滤）

```
指标              FP32                    INT8                     差异
─────────────────────────────────────────────────────────────────────────
RecTime(s)        11.372                  7.722                   +1.47x ★
输出数量           74                      74                       一致
非空文本数         74 (100%)               31 (42%)                -43 ✗✗✗
平均置信度         0.9071                  0.2133                  -0.6938 ✗
文本一致率         0/74 (0%)                                        ✗✗✗
```

### 11.3 失败原因分析

**INT8 量化对 CTC 解码的致命影响：**

```
FP32 CTC 概率分布:
  timestep 1: [blank=0.02, A=0.80, B=0.15, ...]  → 解码出 'A' ✓
  timestep 2: [blank=0.01, X=0.90, Y=0.08, ...]  → 解码出 'X' ✓
  ...

INT8 CTC 概率分布 (量化后):
  timestep 1: [blank=0.45, A=0.30, B=0.20, ...]  → blank 胜出! 吞掉字符 ✗
  timestep 2: [blank=0.60, X=0.25, Y=0.10, ...]  → blank 胜出! 吞掉字符 ✗
  ...
  结果: 大量空白输出 (score≈0)
```

**根因**: PPDetection 的官方 INT8 模型是用 **PaddlePaddle 的 PTQ（训练后量化）** 流程生成的，量化校准集与我们的实际输入分布不匹配，导致：
1. CTC 概率峰值被"抹平"，blank 概率相对上升
2. 低置信度字符全部被 blank 吞噬
3. 即使保留的文本也出现大量错字（如 `'0844谱880其旺'` → `''`）

### 11.4 具体差异示例

| # | FP32 (score) | INT8 (score) | 说明 |
|---|-------------|-------------|------|
| 1 | '0844谱880其旺'(0.971) | ''(0.000) | 整段丢失 |
| 2 | '0080-7郫'(0.469) | ''(0.000) | 整段丢失 |
| 3 | '泺棍米辖提讲影篓季支趵暇球撼茵'(0.991) | ''(0.000) | 高置信度也丢失 |
| 4 | '至轶罢嘤瀘84387-43彤其腿谱腿'(0.988) | '轶罢84387-43彤其腿谱-'(0.811) | 头尾截断+乱码 |
| 5 | '振沧箔沧暇茵'(0.998) | ''(0.000) | 0.998置信度也丢失 |
| 6 | '罢-陟08岱088370838溆'(0.995) | '罢-陟0808830838'(0.853) | 字符级错误 |

### 11.5 结论

| 维度 | 结论 |
|------|------|
| 加速效果 | ★ 有效，1.47x（11.4s → 7.7s） |
| 精度 | ✗✗✗ 不可接受 — 58% 文本变空串，剩余全乱码 |
| 可用性 | ❌ **不可用于生产** |
| 根因 | PaddleOCR 官方 INT8 校准集与实际输入不匹配，CTC 概率被破坏 |

### 11.6 后续可尝试的方向

如果要使用 INT8 量化 Rec，需要：

1. **自定义校准集量化**（推荐）:
   - 用项目实际的 OCR crop 图像作为校准集
   - 使用 ONNX Runtime 的 `quantize_dynamic` 或 `quantize_static`
   - 命令示例：
     ```bash
     python -m onnxruntime.quantization.quantize_static \
       --input ch_PP-OCRv4_rec_mobile.onnx \
       --output rec_custom_int8.onnx \
       --calibration_data_dir ./calib_crops/ \
       --per_channel
     ```

2. **仅量化特定层**（跳过 CTC 输出前的最后几层）:
   - 量化 Conv/BN 层（计算密集型）
   - 保持 Linear 层为 FP32（精度敏感型）

3. **换用其他轻量模型架构**（非量化方向）:
   - SVTR-Lite (PP-OCRv4 新增的 Rec 骨干)
   - 可能原生就比 MobileNet 更快且更准

### 11.7 当前最优配置总结

经过所有测试（GPU/DSP/INT8/线程调优），当前生产配置为：

```python
# ── 最优生产配置 (2026-04-26 最终确认) ──
USE_DLC_DET = False          # Det: ORT-CPU (唯一正确方案)
USE_DLC_CLS = False          # Cls: ORT-CPU (DLC-GPU需batch优化)
REC_ONNX = 'ch_PP-OCRv4_rec_mobile.onnx'  # FP32 (INT8不可用)
intra_op_num_threads = 4     # 已验证ARM最优
REC_BATCH_NUM = 16

# 性能基线 (photo_1.jpg, 74框)
Total: ~15s
  Det:  1.70s  (11%)
  Cls:  0.35s  (2%)
  Rec: 11.40s  (76%)  ← 绝对瓶颈，暂无有效加速方案
  其他:  0.05s  (1%)

# 准确度基线
Texts:     88/88 (100%)
AvgScore:  0.949 (目标≥0.95)
```

---

## 12. ocr_board.py vs copy 2.py 架构对比 (2026-04-26)

> 目的：记录两个版本的差异设计思路，为后续合并优化提供参考。

### 12.1 基本事实

| 维度 | `ocr_board.py` (主文件) | `ocr_board copy 2.py` (副本) |
|------|------------------------|------------------------------|
| 总行数 | **1656** | **1399** (-18.4%) |
| 类名 | `OCRBoardEngine` | `OCRBoardEngine` (相同) |
| 预加载机制 | dict + ort + pyclipper | **完全相同** |
| `det_postprocess` 实现 | pyclipper + shapely + cv2 | **完全相同** |
| `_run_det/cls/rec` 方法 | SDK + ORT 双路径 | **完全相同** |

### 12.2 核心架构差异

#### A. Det 后端策略

```
ocr_board.py:
  USE_DLC_DET = False        # ★ 有开关，可切换
  └── True → multi_tile_det_detect() → 多图块+NMS (~190行专属代码)
  └── False → det_preprocess→_run_det→det_postprocess (ORT单次全分辨率)

copy 2.py:
  无 USE_DLC_DET 配置项      # ★ 彻底移除 DLC Det 分支
  固定走: det_preprocess→_run_det→det_postprocess (ORT单次全分辨率)
  理由: "DLC固定640x640输出限制检测精度"
```

#### B. Cls 后端选择逻辑

```
ocr_board.py:
  USE_DLC_CLS = False       # 显式开关控制
  init_model() 中根据此标志决定 _cls_is_sdk

copy 2.py:
  无显式开关               # 自动探测
  init_model() 中: if _SDK and os.path.exists(.dlc) → 用DLC, 否则ORT-CPU
```

#### C. CLS_THRESH 差异

| 版本 | CLS_THRESH | 含义 |
|------|-----------|------|
| 主文件 | **0.999** | 极度保守，仅极高置信度才旋转（为DLC-GPU误判预留） |
| 副本 | **0.9** | 标准值，ORT-CPU下方向分类本身已够准确 |

### 12.3 设计哲学分野

```
┌─────────────────────────────┐   ┌─────────────────────────────┐
│    ocr_board.py              │   │  ocr_board copy 2.py        │
│    实验平台 / 探索完整版     │   │  生产裁剪 / 决策精简版       │
├─────────────────────────────┤   ├─────────────────────────────┤
│ 设计目标:                   │   │ 设计目标:                    │
│  保留所有探索过的加速路径    │   │  锁定已验证的最优路径         │
│                             │   │                              │
│ 独有能力:                   │   │ 独有能力:                     │
│  ★ 多图块检测系统(~190行)    │   │  ★ 代码量少257行(易维护)     │
│  ★ USE_DLC_DET/CLS 开关     │   │  ★ recognize()无分支(更快)   │
│  ★ DET_MULTI_* 参数组全套    │   │  ★ CLS_THRESH=0.9更实用     │
│  ★ Rec 诊断统计日志          │   │                              │
│  ★ _rect_nms_fast() 快速NMS │   │                              │
│                             │   │                              │
│ 认知负担: 高(需理解双路径)   │   │ 认知负担: 低(线性流程)        │
│ 适用场景: 继续调优/需要灵活性 │   │ 适用场景: 生产部署/稳定运行   │
└─────────────────────────────┘   └─────────────────────────────┘
```

### 12.4 性能影响分析

| 因素 | 主文件影响 | 副本影响 |
|------|-----------|---------|
| `recognize()` 中 `if self._det_is_sdk:` 判断 | 每次1次额外分支（~ns级） | 无此判断 |
| 多图块代码常驻内存 | ~190行字节码（无实际开销，因 `_det_is_sdk=False` 不执行） | 不存在 |
| CLS_THRESH=0.999 vs 0.9 | 更少的180°旋转操作（微幅节省时间） | 更多旋转（但结果正确性一致） |
| **实际推理速度** | **基本相同** | **基本相同** |

### 12.5 两者的共同优点（预加载机制）

两个版本在文件顶部（行34-65）都有相同的预加载模式：

```python
# 1) 重型模块预导入（避免首次调用时的 import 开销）
try:
    import onnxruntime as ort          # ~500ms 的 import 被前置
except ImportError:
    ort = None

try:
    import pyclipper                    # 可选依赖，优雅降级
    from shapely.geometry import Polygon as ShapelyPolygon
    _has_pyclipper = True
except ImportError:
    _has_pyclipper = False

# 2) CTC 字典预加载（避免 init_model() 时等待文件 I/O）
_PRELOAD_DICT = None
with open('ppocr_keys_v1.txt') as f:
    _PRELOAD_DICT = [line.strip() for line in f]
    _PRELOAD_DICT.insert(0, 'blank')
    _PRELOAD_DICT.append(' ')
    while len(_PRELOAD_DICT) < 6625:    # 补齐到模型输出维度
        _PRELOAD_DICT.append('')
```

**这个模式已经存在于两个文件中，不需要额外学习。**

### 12.6 合并建议：如何取两者之长

如果未来要做合并，推荐方案：

```
合并目标版 = 副本的简洁结构 + 主文件的保留能力

1. 默认行为 (99%情况):
   ├── 固定 ORT-CPU (同副本，零分支)
   ├── CLS_THRESH=0.9 (同副本)
   └── recognize() 线性流程 (同副本)

2. 可选能力 (通过环境变量/配置开关激活):
   ├── OCR_USE_DLC=1 → 启用多图块+DLC路径 (从主文件移植)
   ├── OCR_DEBUG=1   → 启用 Rec 诊断日志 (从主文件移植)
   └── 激活时才 import DLC 相关代码 (懒加载，不影响默认性能)

3. 代码组织:
   ├── 基类: OCRBaseEngine (纯ORT-CPU, 线性流程, ~1100行)
   └── 扩展: OCRExtendedEngine(OCRBaseEngine) (添加DLC/多图块能力, ~300行)
```

**核心原则**: 默认路径零开销，高级能力按需激活。

### 12.7 结论

| 问题 | 回答 |
|------|------|
| 成绩差不多吗？ | 是，两者输出完全一致（88框, avgScore≈0.949），速度基本相同 |
| 预加载有差异吗？ | **没有**，两者的预加载机制完全相同（ort/pyclipper/dict 三件套） |
| 思路差异在哪？ | 主文件=实验平台（保留DLC多图块等全部探索成果）；副本=生产裁剪（删除死代码锁定最优路径） |
| 能合并吗？ | 能，建议以副本为基础、按懒加载方式回插主文件的多图块/DLC能力 |
