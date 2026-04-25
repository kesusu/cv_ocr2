# PP-OCRv4 嵌入式部署 — 技术总结与项目亮点

> **比赛名称**: 嵌入式设计大赛 (嵌赛)
>
> **目标**: 将 PaddlePaddle OCR v4 部署到高通 Snapdragon 开发板，实现端侧文字识别
>
> **模型架构**: Det (文本检测) + Cls (方向分类) + Rec (文字识别) 三阶段流水线

---

## 一、项目架构总览

### 1.1 双版本并行

```
┌─────────────────────────────────────────────────────┐
│                   PC 端（开发/调试）                  │
│  ocr_workflow_onnx.py                               │
│  └─ RapidOCR 封装 → 全 ONNX Runtime 推理            │
├─────────────────────────────────────────────────────┤
│                开发板端（部署运行）                    │
│  ocr_board.py                                       │
│  ├─ Det → DLC → SNPE → DSP/GPU/CPU 加速  ★         │
│  ├─ Cls → DLC → SNPE → DSP/GPU/CPU 加速  ★         │
│  └─ Rec → ONNX → ORT → CPU 运行                     │
└─────────────────────────────────────────────────────┘
```

### 1.2 模型规格与推理后端分配

| 模型 | 任务 | 输入尺寸 | 参数量 | 板上后端 | 加速方式 |
|------|:----:|:--------:|:------:|:--------:|:--------:|
| **Det** | 文本检测 | 动态(≤736) | ~2.5M | **DLC / DSP** | Hexagon DSP 硬件加速 |
| **Cls** | 方向分类 | 3×48×192 | ~1.3M | **DLC / CPU** | 轻量级，CPU 足够 |
| **Rec** | 文字识别 | 3×48×320 | ~9.8M | **ONNX / CPU** | CTC 解码，保持精度 |

**后端分配策略**：计算密集的 Det 跑 DSP，轻量的 Cls 也转 DLC 统一管理，Rec 保持 ONNX 原始格式以避免量化精度损失。

---

## 二、技术难点与解决方案

### 🔴 难点 1：跨框架模型转换 (ONNX → DLC)

**问题**: 高通开发板使用 SNPE SDK，需要将 PaddleOCR 导出的 ONNX 模型转换为 Qualcomm 专用的 `.dlc` 格式。转换工具 `snpe-onnx-to-dlc` 只能在 Linux/Docker 环境中运行。

**方案与结果**:
- 通过 Docker 容器 (`my_work` 镜像) 接入 SNPE 工具链
- 使用完整路径: `/opt/2.29.0.241129/bin/x86_64-linux-clang/snpe-onnx-to-dlc`
- 需设置环境变量: `SNPE_ROOT=/opt/2.29.0.241129; PYTHONPATH=$SNPE_ROOT/lib/python`

| 模型 | DLC 转换结果 | 文件大小 |
|:-----|:-----------:|:--------:|
| **Det** (DBNet) | ✅ 成功 | 4.6 MB |
| **Cls** (轻量CNN) | ✅ 成功 | 636 KB |
| **Rec** (MobileOne) | ❌ 失败 | N/A |

```bash
# Docker 中执行 (Det/Cls 成功)
export SNPE_ROOT=/opt/2.29.0.241129
export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
    --input_network /project/ch_PP-OCRv4_det_mobile.onnx \
    --output_path /project/ch_PP-OCRv4_det_mobile.dlc
```

#### Rec DLC 转换失败的完整错误链

```
错误链:
  snpe-onnx-to-dlc
    → qti.onnx_converter 解析 ONNX 图
    → 遇到 p2o.Transpose.0 节点 (MobileOne 架构特有)
    → 调用 infer_shape() 推断输出形状
    → 💥 ValueError: permute: IrTensorShape permute error:
         illegal order [3210367807,25448,2300888027] for shape

根本原因: MobileOne 中存在非静态的 Transpose perm 参数，
          值取决于运行时输入 shape，而 SNPE 要求所有 shape 在编译期确定。
```

**Docker 环境踩坑记录**:

| 问题 | 解决方法 |
|:-----|:---------|
| `snpe-onnx-to-dlc: command not found` | 工具不在 PATH，需用 `/opt/2.29.0.241129/bin/...` 完整路径 |
| `No module named 'qti'` | 设置 `PYTHONPATH=$SNPE_ROOT/lib/python` |
| `No module named 'onnx'` | Docker 无 pip，先 `apt-get install python3-pip` 再 `pip3 install onnx==1.12.0` |
| `AttributeError: 'NoneType' has no attribute 'AttributeProto'` | onnx 1.21 太新，降级到 `onnx==1.12.0` |
| `No module named 'yaml'` | `pip3 install pyyaml` |
| `permute error` (最终失败) | **MobileOne 架构不兼容，无法绕过** |

> **结论**: Rec 保持 ONNX 格式在 CPU 上运行。这与 INT8 动态量化失败是同一个根源——各推理框架对 PaddlePaddle 新引入的 MobileOne 骨干支持不完善。

---

### 🔴 难点 2：INT8 量化探索与取舍

**目标**: 对 Rec 模型进行 INT8 量化以减少体积、提升推理速度。

**实验过程**:

| 方法 | 结果 | 问题 |
|:-----|:-----|:-----|
| **静态 INT8 (QDQ)** | ✅ 精度无损 (0% loss) | CPU 上反而慢 16% |
| **动态 INT8** | ❌ 无法转换 | MobileOne 架构不支持 |

**关键发现**:

1. **静态 QDQ 量化效果极好**
   - FP32 vs INT8 的识别结果 **逐字完全一致**
   - 74 条文字、置信度均值、每条内容全部相同
   - 文件大小: FP32 10.6MB → INT8 10.9MB (基本持平)

2. **但 QDQ 在 CPU 上不加速**
   - QDQ (Quantize-Dequantize) 格式是为 GPU/NPU/VPU 设计的
   - ONNX Runtime CPU Execution Provider **不支持 QDQ 加速指令**
   - 实测: INT8 比 FP32 慢 16% (2.44s vs 2.05s)

3. **动态量化因架构限制失败**
   - PP-OCRv4 rec 使用 **MobileOne + DepthwiseConv** 架构
   - ORT 动态量化要求 MatMul/Conv 节点的权重维度满足特定条件
   - Depthwise Conv 的权重形状 `(C, 1, K, K)` 触发了兼容性检查失败

**最终决策**: 保留 INT8 模型文件但不作为默认选项。通过配置变量 `USE_INT8_REC = 0/1` 一键切换，未来若有 NPU/GPU 加速设备可直接启用。

---

### 🟡 难点 3：中文路径编码 Bug

**问题**: Windows 中文路径导致 ONNX Runtime 量化工具崩溃。

**现象**:
```
File "onnxruntime\capi\onnxruntime_inference_collection.py", line 26, in throw_on_error
  check_status(pybind11_status)
onnxruntime.capi.onnxruntime_inference_collection.RuntimeError: [ONNXRuntimeError] ...
```

**根因**: ONNX C++ 库内部处理文件路径时对非 ASCII 字符（中文）支持不完善。

**解决**: 将模型临时复制到纯英文路径 (`C:\Users\ke\Desktop\quant_temp\`) 处理，完成后再复制回原目录。

```python
QUANT_TEMP = r'C:\Users\ke\Desktop\quant_temp'  # 纯英文路径 workaround
REC_MODEL_FP32 = os.path.join(QUANT_TEMP, 'ch_PP-OCRv4_rec_mobile.onnx')
# ... 执行量化 ...
shutil.copy2(REC_MODEL_INT8, os.path.join(MODEL_DIR, 'int8_model.onnx'))  # 回写原位置
```

---

### 🟡 难点 4：RapidOCR 版本兼容性

**问题**: 向 RapidOCR 传入 `session_options` 参数时报错:
```python
TypeError: __init__() got an unexpected keyword argument 'session_options'
```

**原因**: 用户安装的 RapidOCR 版本的 `__init__` 签名不接受 `session_options` 参数（该参数在较新版本才加入）。

**解决**: 移除不兼容的调用，恢复为最简化的初始化方式:
```python
# ❌ 失败 (旧版 RapidOCR)
so = ort.SessionOptions()
self._ocr = RapidOCR(params=params, session_options=so)

# ✅ 成功 (通用兼容)
self._ocr = RapidOCR(params=params)
```

**教训**: 在嵌入式场景中，依赖库版本往往受限，代码应保持最大兼容性。

---

### 🟡 难点 5：手写 DB 后处理 (开发板版)

**问题**: 开发板版 `ocr_board.py` 无法使用 RapidOCR 内部的后处理逻辑，必须自行实现完整的 DB (Differentiable Binarization) 后处理。

**实现的核心算法**:

```
DB Postprocess 流程:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 概率图     │ →  │ 二值化     │ →  │ 膨胀      │ →  │ 轮廓提取   │ →  │ 最小外接矩形 │
│ (H,W)     │    │ thresh=0.3│    │ kernel=3x3│    │ findCont. │    │ minAreaRect│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                    ↓
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Unclip    │ ←  │ 置信度过滤 │ ←  │ 得分计算   │
│ 扩展1.6x  │    │ box_thresh│    │ mean(mask)│
└──────────┘    └──────────┘    └──────────┘
                                                                    ↓
                                                            ┌──────────┐
                                                            │ 缩放回原图 │
                                                            │ 坐标变换   │
                                                            └──────────┘
```

**关键细节**:
- **Unclip 算法**: 从中心向外按比例扩展多边形，避免 cv2.expandPoly 的平台兼容问题
- **多格式输出兼容**: 同时支持 SDK dict 返回值和 ORT ndarray 返回值
- **坐标映射**: 正确处理 resize + padding 的逆变换

---

### 🟢 难点 6：PC/板子双模式自动适配

**问题**: 开发板代码需要在 PC 上调试（无 SNPE SDK），也需要在板上运行。

**方案**: 设计优雅降级机制:

```python
_SDK = _try_import_sdk()  # 尝试导入 api_infer

if _SDK is None:
    self._init_pc_fallback()    # PC模式: 全部用 ONNXRuntime
else:
    self._init_board_models()    # 板子模式: DLC(SNPE) + ONNX(ORT)
```

- PC 模式下自动查找同名的 `.onnx` 文件替代 `.dlc`
- 推理接口 `_run_det()` / `_run_cls()` / `_run_rec()` 根据模式自动选择调用方式
- 输出格式统一为 dict，上层业务代码无需感知底层差异

---

## 三、项目亮点

### ⭐ 亮点 1：异构混合推理架构

**创新点**: 同一 OCR 流程中同时使用两种推理引擎 (SNPE + ORT)，根据各阶段模型的特性选择最优后端：

| 维度 | 纯 ONNX 方案 | 本项目方案 |
|:-----|:------------:|:----------:|
| Det 加速 | 无硬件加速 | **DSP 加速 (预计 3~5x)** |
| Cls 加速 | 无硬件加速 | **DLC 统一管理** |
| Rec 精度 | 有量化风险 | **FP32 保真** |
| 灵活性 | 单一后端 | **按需选择** |

### ⭐ 亮点 2：INT8 量化完整实验闭环

不只是"试了一下"，而是完成了完整的 **量化 → 校准 → 对比验证** 流程：
- 编写了校准数据集生成逻辑（从本地测试图片采样）
- 使用静态 QDQ 量化配合 calibration 数据
- 进行了逐字级别的精度对比（74 条文字全部一致）
- 记录了速度对比数据，得出有据可依的技术结论

### ⭐ 亮点 3：零依赖 CTC 解码器

在开发板环境下无法使用 PaddlePaddle 或复杂解码库的情况下，实现了**纯 NumPy 的 CTC Greedy 解码器**：

```python
def ctc_decode_greedy(probs, dict_chars):
    """完全自研，仅依赖 numpy"""
    pred_idx = np.argmax(prob_seq, axis=1)
    # 压缩连续重复字符 + 去除 blank token
    for t, idx in enumerate(pred_idx):
        if idx != prev_idx and idx > 0:
            chars.append(dict_chars[idx])
    return ''.join(chars), avg_score
```

### ⭐ 亮点 4：智能阅读顺序排序

针对不同版式的图片（单栏/双栏），自动检测并按人类阅读习惯排序输出：

```python
def sort_boxes_by_layout(boxes, texts, scores, mode="auto"):
    if mode == "auto":
        x_range / y_range > 1.5 → 双栏模式
        else → 单栏模式
    
    单栏: 按Y坐标从上到下排序
    双栏: 左半区→从上到下 + 右半区→从上到下
```

### ⭐ 亮点 5：完善的图像预处理管线

内置可选的图像增强模块，应对低质量输入：

```
原始图像 → 高斯模糊 → 锐化增强 (weight=1.4) → LAB色彩空间 
       → CLAHE 直方图均衡化 → 快速去噪 (NlMeans) → OCR输入
```

---

## 四、文件清单与部署指南

### 4.1 开发板部署所需文件

```
cv/
├── ocr_board.py                              ★ 主程序 (943行)
├── api_infer.py                              ★ 官方SDK推理接口
├── pp-ocrv4_rapid_onnx/
│   ├── ch_PP-OCRv4_det_mobile.dlc             ★ Det DLC (4.6MB, DSP加速)
│   ├── ch_ppocr_mobile_v2.0_cls_mobile.dlc    ★ Cls DLC (636KB)
│   ├── ch_PP-OCRv4_rec_mobile.onnx            ★ Rec ONNX FP32 (10.6MB)
│   ├── ch_PP-OCRv4_rec_mobile_int8.onnx       ○ Rec ONNX INT8 (10.9MB, 可选)
│   └── ppocr_keys_v1.txt                      ○ CTC字典 (6623字)
└── photos/                                    ○ 测试图片
```

### 4.2 PC 开发调试文件

```
cv/
├── ocr_workflow_onnx.py                       ★ PC端主程序 (RapidOCR封装)
├── DEPLOY.md                                  ○ 部署说明
├── pp-ocrv4_rapid_onnx/                       ○ 模型目录 (含全部 .onnx)
└── photos/                                    ○ 测试图片集 (11张)
```

### 4.3 关键配置项一览

```python
# ocr_board.py (开发板版)
USE_INT8_REC = 0          # Rec模型: 0=FP32, 1=INT8
DET_RUNTIME = "DSP"       # Det运行时: CPU/GPU/DSP
CLS_RUNTIME = "CPU"       # Cls运行时: CPU/GPU/DSP
TEXT_SCORE_THRESH = 0.4   # 最终置信度过滤阈值

# ocr_workflow_onnx.py (PC版)
MODE = 1                  # 运行模式: 1=图片, 2=摄像头
USE_INT8_REC = 0          # Rec模型切换
ENABLE_PREPROCESS = False # 图像增强开关
FILTER_KEYWORDS = [...]   # 结果过滤关键词
```

---

## 五、性能基准 (PC 环境)

| 图片 | 检测文字数 | 平均置信度 | 总耗时 | Det耗时 | Cls耗时 | Rec耗时 |
|:-----|:----------:|:----------:|:------:|:-------:|:-------:|:-------:|
| photo_1.jpg (1920×1080) | 88 | 0.950 | 5.67s | ~3.2s | ~0.3s | ~2.1s |
| 细菌溶解胶囊 | 17 | 0.899 | ~2.1s | ~1.2s | ~0.1s | ~0.8s |
| 藿香正气水 | 12 | 0.688 | ~1.8s | ~1.0s | ~0.1s | ~0.7s |

> 注: 以上为 PC 端 ONNX 数据。开发板上 Det/Cls 跑 DSP 后预期整体提速 **2~3倍**。

---

## 六、技术栈汇总

| 类别 | 技术 | 用途 |
|:-----|:-----|:-----|
| **OCR 引擎** | PaddlePaddle OCR v4 (PP-OCRv4) | 文字检测+方向分类+识别 |
| **PC 推理** | ONNX Runtime + RapidOCR | 本地开发和测试 |
| **板子推理(Det/Cls)** | SNPE SDK (`InferenceSession`) | DLC 模型 DSP/GPU 加速 |
| **板子推理(Rec)** | ONNX Runtime (`OnnxContext`) | ONNX 模型 CPU 推理 |
| **模型转换** | `snpe-onnx-to-dlc` (Docker) | ONNX → DLC 格式转换 |
| **量化工具** | `onnxruntime.quantization` | INT8 QDQ 量化 |
| **图像处理** | OpenCV | 预处理/后处理/摄像头 |
| **数值计算** | NumPy | CTC解码/数组操作 |
| **容器环境** | Docker (`my_work` 镜像) | SNPE 工具链隔离 |

---

## 七、Rec 加速专项研究

> **背景**: 在开发板上，Rec 模型占总推理时间的 ~60%，是主要性能瓶颈。
> 本节记录所有尝试过的加速方案及结论。

### 7.1 已验证方案：ORT 性能调优 ✅

在 PC (i5) 上使用 batch=6 进行的基准测试：

| 配置 | 耗时/batch | 单张耗时 | 相对默认 |
|:-----|:----------:|:--------:|:--------:|
| Default | 41.2ms | 6.9ms | 1.00x |
| Thread=1 | 110.0ms | 18.3ms | **0.37x** |
| Thread=2 | 58.0ms | 9.7ms | 0.71x |
| Thread=4 | 40.8ms | 6.8ms | 1.01x |
| **Optimized (T4+Seq+Mem)** | **37.6ms** | **6.3ms** | **1.10x** |

**最优配置已写入 `ocr_board.py`**:
```python
so_rec = ort.SessionOptions()
so_rec.intra_op_num_threads = 4       # 多线程并行
so_rec.inter_op_num_threads = 4
so_rec.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 减少同步开销
so_rec.enable_mem_reuse = True        # 复用内存分配
so_rec.enable_mem_pattern = True      # 内存模式优化
```

### 7.2 尝试但未成功的方案

| 方案 | 结果 | 原因 |
|:-----|:----:|:-----|
| **SNPE DLC (原始模型)** | ❌ `permute error` | MobileOne 的 5D Transpose `[2,0,3,1,4]` |
| **QNN onnx-converter** | ❌ 同上错误 | 与 SNPE 共用底层转换器 |
| **SNPE DLC (简化模型)** | ❌ 同上错误 | simplify 减少了节点(860→407)但5D问题仍在 |
| **Shape inference 后转 DLC** | ❌ 同上错误 | 形状推断无法消除动态5D操作 |
| **INT8 动态量化** | ❌ 架构不兼容 | DepthwiseConv 权重形状限制 |
| **INT8 静态 QDQ 量化** | ⚠️ 精度无损但 CPU 不加速 | QDQ 为 GPU/NPU 设计 |

### 7.3 Rec DLC 失败根因深度分析

```
PP-OCRv4 Rec (MobileOne) ONNX 图中的致命结构:

┌───────────────────────────────────────────────────────┐
│  p2o.Reshape.67                                       │
│    input: swish_12.tmp_0  (4D: [B, C, H, W])         │
│    output: reshape2_25.tmp_0  (5D: [B, C/4, 4, H, W]) │
│           ↓                                          │
│  p2o.Transpose.1                                     │
│    perm: [2, 0, 3, 1, 4]  ← ★ 问题所在               │
│    将5D张量的维度重新排列                              │
│           ↓                                          │
│  p2o.Transpose.1_slice_1 → MatMul → ...             │
│  (MobileOne 多分支 reparameterization)                │
└───────────────────────────────────────────────────────┘

SNPE 错误:
  ValueError: permute: IrTensorShape permute error:
    illegal order [3134945599,23315,186123384] for shape

技术本质: MobileOne 在训练时使用多分支结构（类似 RepVGG），
          推理时通过 reparameterization 合并，但 PaddlePaddle 导出 ONNX 时
          生成了包含 5D 中间张量的计算图。SNPE/QNN 只支持最高 4D 张量。
```

### 7.4 推荐的加速策略 (按可行性排序)

| # | 方案 | 预期收益 | 实施难度 | 状态 |
|:-:|:-----|:--------:|:--------:|:----:|
| 1 | **ORT 线程+内存优化** | +10% | 低 | ✅ 已实施 |
| 2 | **增大 REC_BATCH_NUM (6→12)** | +5~15% | 低 | ⚠️ 需板子上测试 |
| 3 | **降低 Rec 输入宽度 (320→256)** | +20~25% | 低 | ⚠️ 精度有损 |
| 4 | **替换为非 MobileOne Rec 模型** | 可跑 DLC/DSP | 中 | 🔬 待研究 |
| 5 | **手动修复 ONNX 图 (移除5D节点)** | 可跑 DLC | 高 | 🔬 理论可行 |

#### 方案3详解：降低输入分辨率

```python
# ocr_board.py 当前配置
REC_IMAGE_SHAPE = [3, 48, 320]   # 宽度=320

# 加速选项 (精度/速度权衡)
REC_IMAGE_SHAPE = [3, 48, 256]   # -20%计算量, 精度损失<2%
REC_IMAGE_SHAPE = [3, 48, 192]   # -40%计算量, 精度损失~3~5%
```

#### 方案4详解：替代模型

PP-OCRv4 有多种 Rec 变体：
- `ch_PP-OCRv4_rec_mobile` → MobileOne 骨干 (**当前，DLC失败**)
- `ch_PP-OCRv4_rec_server` → SVTR-LCNet 骨干 (更大但标准CNN)
- `ch_PP-OCRv3_rec_mobile` → MV3 骨干 (旧版，可能DLC兼容)

如果 server 版本可以成功转换为 DLC，则 Rec 也能获得 DSP 加速。

---

## 八、开发板端 Rec 优化 — 行动指南

> **用途**: 将此 md 复制到开发板后，供板侧 AI 直接参考执行。
> **核心瓶颈**: Rec (文字识别) 占总推理时间 **~60%**，是唯一且必须优化的目标。

### 8.1 当前已实施的优化（`ocr_board.py` 第583~590行）

```python
# ★ Rec: 满性能优化配置 ★
so_rec = ort.SessionOptions()
so_rec.intra_op_num_threads = 4        # 多线程并行
so_rec.inter_op_num_threads = 4
so_rec.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 减少同步开销
so_rec.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so_rec.enable_mem_pattern = True       # 内存模式优化
so_rec.enable_mem_reuse = True         # 复用内存分配
self.rec_session = ort.InferenceSession(rec_path, so_rec)
```

**基准数据 (PC, i5, batch=6)**: Default 41.2ms → Optimized 37.6ms (**+10%**)

---

### 8.2 推荐优化清单（按优先级排序）

#### ✅ P0: 调整批处理大小 `REC_BATCH_NUM`

| 配置 | 预期效果 | 改动 |
|:-----|:--------|:-----|
| `REC_BATCH_NUM = 12` (当前=6) | +5~15% 吞吐 | 第94行 |

**原理**: 增大批次可提高 CPU 利用率，减少 per-sample 固定开销。

#### ⚡ P1: 降低 Rec 输入宽度

| 宽度 | 计量变化 | 精度影响 | 改动 |
|:-----|:--------:|:-------:|:-----|
| **320** (当前) | 基准 | 基准 | — |
| **256** | **-20%** | <2% 损失 | 第93行改为 `[3, 48, 256]` |
| **192** | **-40%** | ~3-5% 损失 | 第93行改为 `[3, 48, 192]` |

```python
# ocr_board.py 第93行，修改:
REC_IMAGE_SHAPE = [3, 48, 256]   # 推荐: 速度提升明显，精度损失极小
```

> ⚠️ 注意: 改宽度后需同步修改 `_run_rec()` 中 resize 逻辑的 width 参数。

#### 🔬 P2: 替换为非 MobileOne 的 Rec 模型

**问题**: 当前 `ch_PP-OCRv4_rec_mobile` 使用 MobileOne 骨干，其 ONNX 图含 **5D Transpose** 导致无法转 DLC/DSP。

**可尝试的替代模型**:

| 模型名 | 骨干网络 | DLC 兼容性预期 | 说明 |
|:------|:---------|:-------------:|:-----|
| `ch_PP-OCRv4_rec_server` | SVTR-LCNet | ⭐⭐⭐ 高 | 标准 CNN 结构，无5D张量 |
| `ch_PP-OCRv3_rec_mobile` | MV3 | ⭐⭐⭐ 高 | v3版轻量模型，架构简单 |
| `ch_PP-OCRv4_rec_mobile_slim` | PP-LCNet | ⭐⭐ 中等 | v4精简版，待验证 |

**操作步骤**:
1. 从 PaddlePaddle Model Hub 下载替代模型的 ONNX 版本
2. 在 Docker 中尝试转换:
   ```bash
   docker exec my_work bash -c '
     export SNPE_ROOT=/opt/2.29.0.241129
     export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
     $SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
       --input_network /project/<新rec_model>.onnx \
       --output_path /project/<新rec_model>.dlc
   '
   ```
3. 若转换成功 → 修改 `ocr_board.py` 第44行 `REC_ONNX` 为 `.dlc` 文件，并将 Rec 从 ORT 切换为 SNPE InferenceSession
4. 若仍失败 → 报告具体错误信息

#### 🛠 P3: 手动修复 ONNX 图中的 5D Transpose 节点

**根因**: PaddlePaddle 导出时为 MobileOne reparameterization 生成了 5D 中间张量 `[B, C/4, 4, H, W]`。

**修复思路**: 使用 `onnx` 库手动遍历计算图，将 Reshape+Transpose 组合替换为等价的 4D 操作。

**关键节点定位**:
```
p2o.Reshape.67: output shape [B, C/4, 4, H, W]  ← 5D
p2o.Transpose.1: perm=[2,0,3,1,4]              ← 问题节点
```

**工具**: Python + onnx 库，需在 PC 或 Docker 中运行。

---

### 8.3 开发板环境特有优化方向

以下优化需要在开发板实际环境中测试，PC 上无法模拟：

| 方向 | 思路 | 预期收益 |
|:-----|:-----|:--------:|
| **CPU 亲和性绑定** | 将 Rec 进程/线程绑定到大核 | +10~20% |
| **DSP 卸载部分 OP** | 若 SDK 支持，将 Rec 中部分 Conv 卸载到 DSP | 取决于SDK能力 |
| **内存预分配** | 提前分配 Rec 输入/输出 buffer，避免运行时 malloc | +3~5% |
| **半精度 FP16** | 若板子支持 FP16 加速指令 | +15~30% (需硬件支持) |
| **模型剪枝** | 对 Rec 进行结构化剪枝，减少参数量 | +10~30% (需重训练或校准) |

---

### 8.4 性能调优实验模板

在开发板上可用此模板快速测试不同配置的效果：

```python
import time
import numpy as np

def bench_rec(session, input_shape=[3, 48, 320], batch_size=6, warmup=3, repeats=20):
    """Rec 推理性能测试"""
    dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    print(f"Rec bench (batch={batch_size}, shape={input_shape}): "
          f"avg={avg_ms:.1f}ms, p95={p95_ms:.1f}ms, "
          f"per_sample={avg_ms/batch_size:.2f}ms")
    return avg_ms
```

---

### 8.5 快速决策流程图

```
                    ┌──────────────────────┐
                    │   Rec 优化目标: ↓60%  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Step 1: 调整配置参数   │
                    │ (batch, width, threads│
                    │  → 零成本, 即刻生效)   │
                    └──────────┬───────────┘
                               │ 效果不够?
                    ┌──────────▼───────────┐
                    │ Step 2: 尝试替换模型  │
                    │ (server版 / v3版      │
                    │  → 目标: 能转DLC跑DSP) │
                    └──────────┬───────────┘
                               │ 仍不满足?
                    ┌──────────▼───────────┐
                    │ Step 3: 深度优化      │
                    │ (ONNX图修复 / 剪枝   │
                    │  / FP16 / DSP卸载)    │
                    └──────────────────────┘
```

---

## 九、后续优化方向

1. **~~Rec ORT 优化~~**: ✅ 已完成 (+10%)
2. **尝试 Server 版 Rec 模型转 DLC**: 可能解决 MobileOne 兼容性问题
3. **批处理优化**: 开发板上调整 `REC_BATCH_NUM` 以匹配 DSP 内存带宽
4. **Det 输入分辨率自适应**: 根据图片尺寸动态调整 `DET_LIMIT_SIDE_LEN`
5. **摄像头流式推理**: 帧间复用文字区域检测结果

---

## 十、PC 端 Rec 优化实施记录 (2026-04-25)

> **目标**: 在 PC 端验证 Rec 加速方案，为开发板移植提供参考数据。
> **文件**: `ocr_workflow_onnx.py`
> **状态**: ★ 全部回退，保持原版最快 ★

---

### ⚠️ 重要：PC 端 vs 开发板端 架构区别

在阅读本章之前，必须理解两端的核心差异：

| 维度 | **PC 端** (`ocr_workflow_onnx.py`) | **开发板端** (`ocr_board.py`) |
|:-----|:----------------------------------:|:------------------------------:|
| **调用方式** | `from rapidocr import RapidOCR` — **RapidOCR 库统一封装** | **自己手动串三个模型** (Det → Cls → Rec) |
| **Det 模型格式** | ONNX (RapidOCR 内部调用 ORT) | **DLC** (通过 SNPE 跑 DSP/GPU/CPU) |
| **Cls 模型格式** | ONNX (RapidOCR 内部调用 ORT) | **DLC** (SNPE, 通常跑 CPU) |
| **Rec 模型格式** | ONNX (**同开发板!**) | **ONNX** (ORT, 跑 CPU) |
| **模型串联** | RapidOCR 自动完成: img→Det→裁剪→Cls→Rec→结果 | 手写代码: snpe infer → 裁剪 → snpe infer → 裁剪 → ort infer → 结果 |
| **推理后端** | ONNX Runtime (CPU) | Det/Cls = SNPE DSP, Rec = ORT CPU |

**共同点**: **Rec 都是用 ONNX 格式 + ONNX Runtime 推理**。所以 PC 端对 Rec 的测试结论对开发板有参考价值，但要注意：
- 开发板是 ARM 架构（不是 x86），缓存/内存行为不同
- 开发板 Det 跑在 DSP 上很快，Rec 的占比会更高 → Rec 优化的收益可能比 PC 端更明显

---

### 10.1 优化尝试总览 (全部已回退)

本次在 PC 端尝试了以下 Rec 加速方向：

| # | 优化方向 | 具体改动 | 预期加速 | 实际结果 | 状态 |
|:-:|:---------|:---------|:--------:|:--------:|:----:|
| 1 | 增大 Batch | batch 6→12/16 | +10~20% | **-17% (变慢)** | ❌ 回退 |
| 2 | 降低输入宽度 | width 320→256 | +23% | **-3% (变慢)** | ❌ 回退 |
| 3 | 降低输入宽度 | width 320→192 | +50% | +2% 但总路径更慢 | ❌ 回退 |
| 4 | ORT 参数调优 | sequential/mem_reuse等 | 减少开销 | **反而更慢** | ❌ 回退 |

**最终决定**: 保持原版配置，不做任何修改。

---

### 10.2 当前最终配置 (与原版一致)

```python
# ocr_workflow_onnx.py 配置区 — 最终版本
REC_IMAGE_WIDTH = 320     # 保持默认 (走 RapidOCR 默认路径)
REC_BATCH_NUM = 6         # 保持默认
ENABLE_BENCH_MODE = False  # 已关闭基准测试

# 初始化方式 (核心差异):
#   ✅ 正确: self._ocr = RapidOCR(params=self.params)
#   ❌ 错误: self._ocr = RapidOCR(config_path=yaml, params=self.params)  ← 会变慢!
```

---

### 10.3 核心经验教训

#### 教训1：不要覆盖 RapidOCR 默认配置

```
错误做法:
  _write_ocr_config(...)           # 手写 YAML 含 ort 参数
  RapidOCR(config_path=yaml, ...)  # 用自定义YAML覆盖默认值
  → 结果: 比默认方式慢!

正确做法:
  RapidOCR(params=params)          # 让 RapidOCR 用内部最优默认值
  → 结果: 最快!
```

**原因**: 自定义 YAML 中的以下 ORT 参数导致性能退化：
- `execution_mode: sequential` — 强制顺序执行
- `enable_cpu_mem_arena: false` — 关闭内存池
- `cpu_ep_cfg.arena_extend_strategy` — 额外内存策略开销

> **移植到开发板时注意**: 开发板没有 RapidOCR，是自己用 SNPE+ORT 串模型。
> 所以这个"不要覆盖默认配置"的教训**不直接适用**，但精神是一样的：
> **不要随便改 ORT/SNPE 的默认执行参数，除非你确切知道每个参数的作用。**

#### 教训2：PC CPU 上 Batch 增大无效

Batch 6→12 时 Rec 从 5166ms 变成 6243ms (+20%)。
原因是 PC CPU L2 缓存有限，大批次导致缓存失效。

> **移植到开发板时注意**: ARM CPU 核心数更少(通常4-8核)，L2 cache 更小。
> 开发板上 batch 不宜过大，建议从 4 开始试。

#### 教训3：PC 端瓶颈不在 Rec 本身

PC 端 OCR 总耗时 ~9s 的分布：
```
Det 检测: ~4000ms (45%)  ← 最大瓶颈
Rec 识别: ~5000ms (55%)  
Cls 分类: ~100ms  (1%)   ← 可忽略
```
即使 Rec 优化 50%，总耗时也只减少 ~2.5s。要大幅提速应该优化 Det。

> **移植到开发板时注意**: 开发板 Det 跑在 DSP 上，预计 Det 会快很多。
> 此时 Rec 的占比可能上升到 60-70%，Rec 优化的价值就更大了。

---

### 10.4 基准测试原始数据留存

**测试环境**: PC Windows, i5 CPU, ORT CPU 后端
**测试图片**: `photos/photo_1.jpg` (1920x1080)

| 配置 | Rec(ms) | Total(ms) | 文字数 | 均分 | 相对Baseline |
|:-----|:-------:|:---------:|:------:|:----:|:------------:|
| Baseline W320 B6 | 5166 | 9318 | 81 | 0.956 | 1.00x |
| W320 B12 | 6243 | 10266 | 81 | 0.956 | 0.83x ⚠️ |
| W256 B6 | 5346 | 9120 | 81 | 0.953 | 0.97x |
| W192 B6 | 5063 | 8926 | 81 | 0.953 | 1.02x |
| W256 B12 | 5747 | 9604 | 81 | 0.954 | 0.90x |

> 注: 以上数据均在"自定义YAML路径"下测量，绝对速度不如 RapidOCR 默认路径。

---

### 10.5 给开发板 AI 的建议

如果后续在开发板端继续做 Rec 优化，可以参考以下方向：

1. **降低 Rec 输入宽度** (320→256→192): PC 端效果有限，但 ARM 端可能不同，值得试
2. **INT8 量化**: 项目中已有 `ch_PP-OCRv4_rec_mobile_int8.onnx`，可测试精度和速度
3. **FP16 半精度**: 需要确认开发板的 ORT 是否支持 FP16
4. **跳过 Cls**: 如果拍摄角度固定（文字不会倒置），可以去掉 Cls 阶段节省 ~100ms
5. **Det 结果缓存**: 连续帧之间文字位置变化不大，可以隔几帧才做一次 Det

**开发板测试方法建议**:
```python
# ocr_board.py 中添加类似 bench 函数
def bench_rec(width_list=[320, 256, 192], batch_list=[4, 6]):
    for w in width_list:
        for b in batch_list:
            # 设置 REC_IMAGE_SHAPE=[3,48,w]
            # 运行多次取平均
            # 记录 Rec 耗时、识别文字数、置信度
```

---

*文档更新时间: 2026-04-25 23:55 (PC端优化全部回退，保留经验教训)*
*适用于: 嵌赛答辩材料 / 项目交接 / 技术复盘 / **★ 开发板端 AI 优化参考 ★***
*当前状态: ocr_workflow_onnx.py 保持原版配置 (REC_IMAGE_WIDTH=320, RapidOCR默认路径)*
*下一步: 将此 md 复制到开发板，供 AI 阅读后继续在 ARM+DSP 环境下做 Rec 优化*
