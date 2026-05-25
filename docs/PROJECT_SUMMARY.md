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

## 十一、开发板端实测最终结论 (2026-04-26)

> **来源**: `板端数据/` 目录下的全部调试日志 (22个脚本 + 详细md)
> **环境**: QCS6490 Snapdragon 开发板, fiboaisdk (SNPE), ARM 8核 CPU, Adreno GPU
> **测试图片**: photo_1.jpg (1920x1080), RapidOCR 基准: **88文本框, avg=0.950**
> **最终结论**: 全 ORT-CPU 是当前唯一正确方案 — 精度完美匹配基准

---

### 11.1 计划 vs 实际 — 架构重大修正

| 维度 | **计划 (第一~九章写的)** | **实际 (板上实测结果)** |
|:-----|:------------------------:|:----------------------:|
| Det 后端 | **DLC → DSP 硬件加速** | ❌ DSP/GPU/CPU 输出固定 640×640, 仅12框 |
| Cls 后端 | **DLC → 统一管理** | ❌ DLC-GPU 系统性误旋转20/88框, 损失19%文本 |
| Rec 后端 | ONNX → ORT-CPU | ✅ 与计划一致 (MobileOne无法转DLC) |
| **最终方案** | 异构混合推理 (SNPE+ORT) | **全 ORT-CPU 多线程** |

**根因总结**:
- **Det DLC**: 编译期固化输出 shape=640×640, 高分辨率输入时概率被稀释。多图块检测调到88框但坐标偏移1-5px导致Rec成功率仅38%
- **Cls DLC**: 存在未知的精度损失(INT8量化? FP16?), 对20个框错误判断为ROT180(置信度0.906~0.998), 导致文字倒置后Rec输出blank token
- **Rec INT8(官方模型)**: 校准集与本项目输入不匹配, CTC概率被破坏, 74条中58%变空串, 剩余全乱码

---

### 11.2 各方案实测对比总表

| # | Det后端 | Cls后端 | Rec后端 | Texts | AvgScore | 总耗时 | 状态 |
|:-:|:-------:|:-------:|:-------:|:-----:|:--------:|:------:|:----:|
| 1 | ORT-CPU | ORT-CPU | ORT-FP32 | **88** | **0.949** | **12.95s** | ★★★ **最优** |
| 2 | ORT-CPU | DLC-GPU | ORT-FP32 | 71 | 0.928 | ~14s | ✗ Cls误旋转 |
| 3 | DLC-CPU(多图块) | SDK-CPU | ORT-FP32 | ~15 | 0.65* | ~28s | ✗ Rec质量差 |
| 4 | DLC-GPU | — | — | 0-8 | <0.20 | — | ✗ 全噪声 |
| 5 | ORT-CPU | ORT-CPU | INT8(官方) | 31 | 0.213 | ~8.5s | ✗ 精度崩 |
| 6 | RapidOCR(PC基准) | ORT-CPU | ORT-FP32 | 88 | 0.950 | ~14s | ✓ 参考线 |

> *注: 方案3的 score 是检测置信度非 Rec 分数

---

### 11.3 最终生产配置与性能基线

```python
# ocr_board.py 最终配置 (2026-04-26 确认)
USE_DLC_DET = False              # Det: ORT-CPU (动态分辨率, 高精度)
USE_DLC_CLS = False              # Cls: ORT-CPU (零误旋转)
REC_ONNX = 'ch_PP-OCRv4_rec_mobile.onnx'  # FP32 (官方INT8不可用)
intra_op_num_threads = 4          # ARM 多核 (已验证最优)
REC_BATCH_NUM = 16
DET_LIMIT_SIDE_LEN = 736
DET_LIMIT_TYPE = 'min'
DET_THRESH = 0.20
DET_BOX_THRESH = 0.35
CLS_THRESH = 0.90
TEXT_SCORE_THRESH = 0.4
```

**性能分布 (photo_1.jpg, 1920×1080)**:

| 阶段 | 耗时 | 占比 | 说明 |
|:----:|:----:|:----:|:-----|
| Det | 1.70s | 13% | ORT-CPU, 动态分辨率 |
| Cls | 0.31s | 2% | ORT-CPU, batch=55~88 |
| Rec | 10.86s | **84%** | ORT-CPU, batch=16, **绝对瓶颈** |
| 其他 | 0.08s | 1% | 预处理/后处理 |
| **总计** | **~12.95s** | **100%** | **88/88文本, avg=0.949** |

---

### 11.4 核心经验教训 (精简版)

#### 教训1: 硬件加速 ≠ 一定更快更好
DSP/GPU 加速需满足: 模型适合硬件、I/O shape匹配、数值精度满足、batch支持。
本项目三个条件均不满足 → ORT-CPU 多线程反而是最优解。

#### 教训2: DLC 编译参数决定一切, 且不可逆
ONNX→DLC 转换一旦固化输出 shape, 运行时无法改变。转换前必须确认动态 shape 行为。

#### 教训3: 端到端精度 > 单模块速度
Det从88→12框、Cls导致71文本、Rec 58%空串 — 过度追求单模块加速的代价是无法接受的。

#### 教训4: PC端优化结论不能直接照搬到ARM端
- PC端: 增大Batch反而慢17%(L2缓存限制)、改YAML配置反而慢
- ARM端: 可能完全不同(核少cache小), 但本次因DLC精度问题未能验证

---

### 11.5 后续真正可行的优化方向

当前瓶颈明确: **Rec 占 84% (10.86s)**。以下是按可行性排序的方向:

| # | 方向 | 思路 | 预期收益 | 难度 | 备注 |
|:-:|:-----|:-----|:--------:|:----:|:-----|
| P0 | 增大 REC_BATCH_NUM | 16→32 | Rec -15~25% | 极低 | 纯配置修改 |
| P1 | 降低 Rec 输入宽度 | 320→256 | Rec -20% | 低 | 精度损失<2%, ARM端待验证 |
| P2 | 替换非MobileOne Rec | v3(MV3)/server(SVTR) | 可能跑DLC+DSP | 中 | 需重新转换模型 |
| P3 | CPU亲和性绑定 | taskset绑大核 | 整体+10~20% | 低 | ARM平台特有 |
| P4 | 跳过Cls | 文字不倒置时可省 | 省0.31s | 低 | 取决于拍摄角度固定性 |
| P5 | 自定义校准集INT8量化 | 用项目实际图片做校准 | Rec可能+30~50% | 中 | 需要PC端工具链 |

> ⚠️ 第八章/第九章的理论建议(P0 batch增大/P1宽度降低)在**PC端已验证无效或效果极微**,
> 但在**ARM+全ORT-CPU环境**下值得重新验证, 因为瓶颈分布不同(Rec占84% vs PC端55%)。

---

### 11.6 详细调试日志索引

以下文件包含完整的调试过程、参数搜索数据、逐框分析结果:

```
板端数据/
├── PROJECT_SUMMARY.md              ← 完整版技术文档 (1256行, 12章)
│   第十章: 开发板实际部署测试 (SDK适配/DLC固定输出/Cls batch问题)
│   第十一章: 待解决问题与改进计划
│   第十二章: GPU 硬件加速改造 (DLC-GPU 尝试)
│
└── Log/
    ├── dlc_det_optimization_log.txt  ← ★ 核心日志: 多图块检测 + Cls误分类根因
    │   Phase 1-4: Det DLC多图块参数搜索 (22个脚本, 4轮网格扫描)
    │   Phase 5:   ★ Cls DLC-GPU 误旋转根因诊断 (20/88框证据表)
    │   最终方案:  全ORT-CPU对比总表
    │
    ├── dlc_gpu_acceleration_guide.md ← DLC-GPU 加速指南
    └── [其余 20 个 .txt 日文]          ← 各阶段原始输出/中间分析
```

**调试脚本清单 (22个 `_debug_*.py`)**:
- Phase 1: `_debug_dlc_system.py` — DLC vs ORT 基准对比
- Phase 2: `_debug_optimize{,_v2}.py` — 多图块+NMS实验
- Phase 3: `_debug_fast{,_v2,_v3}.py` + `_debug_finetune.py` + `_debug_score_scan.py` + `_debug_precise.py` — 参数网格搜索
- Phase 4-5: `_debug_gap.py`, `_debug_missing.py`, `_debug_det_params.py`, `_debug_diagnose.py`, `_debug_box_detail.py`, `_debug_crop_diag.py`, `_debug_rec_raw.py`, `_debug_box_compare.py`, `_debug_cls_culprit.py`, `_debug_cls_vs_ort.py` — 根因诊断链
- 其他: `_debug_rec_int8.py`, `_debug_rec_thread.py`, `_debug_cls_gpu_recovery.py`

---

---

## 十二、EasyOCR ONNX 探索记录 (2026-04-27)

> **目标**: 评估高通 AI Hub 预编译的 EasyOCR 模型是否可作为 PaddleOCR 的替代方案
> **结论**: ❌ **不可行** — 预编译模型为英文专用(97类)，中文版导出失败
> **最终选择**: 继续使用 PaddleOCR v4 (PP-OCRv4) 作为唯一 OCR 引擎

---

### 12.1 探索背景与动机

| 动机 | 说明 |
|:-----|:-----|
| **备选方案** | 若 PaddleOCR Rec 无法加速，考虑换用 EasyOCR |
| **官方支持** | 高通 AI Hub 提供预编译 EasyOCR ONNX/DLC 模型 |
| **多语言** | EasyOCR 原版支持 80+ 语言（含中文 ch_sim） |

---

### 12.2 技术探索过程

#### Step 1: 获取预编译模型

从高通 AI Hub / HuggingFace 下载了两个版本：

```
easyocr-onnx-float/easyocr-onnx-float/
├── detector.onnx          (CRAFT 文本检测器, ~5.9MB)
├── recognizer.onnx        (CRNN 文本识别器, ~11MB)
└── metadata.json

easyocr-onnx-w8a8/easyocr-onnx-w8a8/
├── detector.onnx          (w8a8 量化版, IR v12)
├── recognizer.onnx        (w8a8 量化版, IR v12)
└── metadata.json
```

#### Step 2: Python 3.8 兼容性障碍

| 问题 | 解决方案 | 结果 |
|:-----|:---------|:-----|
| onnxruntime 不支持 IR v12 (最高 v10) | 将 IR version 从 12 降到 10 | ✅ 可加载 |
| w8a8 QDQ 格式输出为 uint8 | 尝试反量化 → 去量化转 float | ⚠️ 图结构破坏 |
| 最终方案 | 放弃 w8a8，改用 float 版本 | ✅ 正常运行 |

#### Step 3: Float 模型跑通

编写 `test_easyocr_w8a8.py` 完整测试脚本（~420行），实现：

```
EasyOCR ONNX 推理流程:
┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│ 原始图像   │ →  │ CRAFT Detector │ →  │ 后处理     │ →  │ 文本区域   │
│ RGB       │    │ (ONNX float)  │    │ getDetBoxes│   │ 裁剪      │
└──────────┘    └──────────────┘    └──────────┘    └────┬─────┘
                                                           ↓
┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│ CTC解码   │ ←  │ CRNN Recognizer│ ←  │ 灰度+标准化 │ ←  │ 灰度转换   │
│ Greedy    │    │ (ONNX float)  │    │ (x-128)/128│   │ resize   │
└──────────┘    └──────────────┘    └──────────┘    └──────────┘
```

**实测结果 (photo_1.jpg, 药品说明书)**:

| 指标 | EasyOCR float (ONNX) | PaddleOCR v4 (FP32) |
|:----:|:---------------------:|:-------------------:|
| 检测文本数 | 2 条 (`"Z("`, `"ZFC"`) | **74 条** |
| 耗时 | **0.61s** | 5.28s |
| 中文识别 | ❌ 全部乱码 | ✅ 完美 |

#### Step 4: 根因分析 — 英文专用模型

```python
# metadata.json 关键信息:
"input_image": { "shape": [1, 3, 608, 800], "dtype": "float32" }
"output_preds": { "shape": [1, 199, 97] }     # ★ 97 类 = ASCII 字符集
# easyocr_md/model.py 第36行:
LANG_LIST = ["en"]  # ★ 硬编码英文
```

**97 类字符表**: 0=blank + 96 个可打印 ASCII 字符 (0-9, a-z, A-Z, 标点)
**中文需要**: ~5000+ 类 (简体中文字典)

---

### 12.3 中文版导出尝试

| 方法 | 结果 | 失败原因 |
|:-----|:----:|:---------|
| 高通 AI Hub `--lang ch_sim` | ❌ 未找到现成下载 | 只有英文预编译包 |
| EasyOCR 库导出 ONNX (Detector) | ✅ 成功 | CRAFT 无动态 shape 问题 |
| EasyOCR 库导出 ONNX (Recognizer) | ❌ 失败 | AdaptiveAvgPool2d 动态尺寸不支持静态导出 |
| EasyOCR 库直接推理 (跳过ONNX) | ✅ 可用 | 但无法部署到开发板 |

#### EasyOCR 中文直接推理结果:

| 指标 | EasyOCR 中文 (ch_sim+en) | PaddleOCR v4 (FP32) |
|:----:|:------------------------:|:-------------------:|
| 检测文本数 | **136 条** (大量碎片化) | 74 条 (干净) |
| 耗时 | 12.91s | **5.28s** |
| 低分结果占比 | >50% (score < 0.1) | <5% (score > 0.9) |

---

### 12.4 最终结论

```
┌─────────────────────────────────────────────────────────┐
│                  EasyOCR 可行性评估                      │
├──────────────┬──────────────────────────────────────────┤
│ 预编译 ONNX  │ ❌ 仅英文(97类), 中文不可用               │
│ 中文 ONNX 导出│ ❌ CRNN 含动态 OP, 无法静态导出           │
│ 直接库调用   │ ⚠️ 能用但慢(12.9s), 且无法部署到板端      │
│ vs PaddleOCR │ ❌ 全面落后: 精度/速度/部署灵活性均不如    │
├──────────────┴──────────────────────────────────────────┤
│  结论: 放弃 EasyOCR, 继续优化 PaddleOCR v4              │
└─────────────────────────────────────────────────────────┘
```

### 12.5 经验教训

1. **预编译模型不等于完整方案** — 高通 AI Hub 的 EasyOCR 只打了英文子集
2. **IR 版本兼容性** — onnxruntime 1.x 在 Python 3.8 上只支持 IR ≤10，新模型需降级或升级运行时
3. **QDQ 量化格式** — w8a8 模型的 QuantizeDequantize 节点在降级后容易破坏图结构
4. **CRAFT 后处理复杂度** — 文字检测的后处理（score_map → text box）比 DBNet 复杂得多
5. **字符表决定一切** — 97 类 vs 6623 类，模型能力天差地别

### 12.6 产出物清单

| 文件/目录 | 说明 | 保留? |
|:----------|:-----|:-----:|
| `test_easyocr_w8a8.py` | EasyOCR ONNX 完整推理脚本 (~420行) | ✅ 保留供参考 |
| `easyocr-onnx-float/` | 英文 float 模型 | ⚠️ 可删(仅英文) |
| `easyocr-onnx-w8a8/` | 英文 w8a8 量化模型 | ⚠️ 可删(仅英文) |
| `easyocr_md/` | 高通 AI Hub 例程代码 | ✅ 保留(有参考价值) |
| `_export_chinese_easyocr.py` | 中文模型导出脚本 | ❌ 临时文件 |
| `_test_easyocr_ch.py` | EasyOCR 中文直接推理 | ❌ 临时文件 |
| `_check_quant.py` | 量化参数检查 | ❌ 临时文件 |
| `_dequant.py` | 去量化脚本 | ❌ 临时文件 |
| `_downgrade_onnx.py` | IR 降级脚本 | ❌ 临时文件 |
| `_run_paddle.py` | PaddleOCR 对比脚本 | ❌ 临时文件 |
| `_fix.py` | f-string 修复 | ❌ 临时文件 |
| `_ch_test_out.txt` | 中文测试输出 | ❌ 临时文件 |
| `_paddle_out.txt` | Paddle 输出 | ❌ 临时文件 |

---

*文档更新时间: 2026-04-27 (EasyOCR 探索完结)*
*适用于: 嵌赛答辩材料 / 项目交接 / 技术复盘*

---

## 十三、摄像头拍摄 + OCR 预处理联合优化 (2026-04-29)

> **背景**: 使用 IMX577 摄像头 (1200万像素, USB2.0) 拍摄药品说明书后进行 OCR 识别，
> 发现过曝问题、预处理参数导致文字识别错误等问题。
> **结论**: 完成摄像头参数清理、自适应亮度校正、预处理管线调优三大优化。

---

### 13.1 摄像头过曝事件与修复

#### 问题现象

使用 `camera_test.py` 拍摄的图片呈现**一片全白**（mean > 250），不仅程序调用如此，Windows 系统相机应用也过曝。

#### 根因分析

项目中 **5 个文件**都设置了会与摄像头自动曝光 (AE) 冲突的硬件参数：

| 文件 | 危险参数 | 影响 |
|:-----|:---------|:-----|
| `camera_test.py` | `brightness=160, exposure=-3, gain=100` | 最直接触发 |
| `ocr_workflow_onnx.py` | 同上 | 每次运行持续写入 |
| `ocr_workflow_onnx copy.py` | 同上 | 同上 |
| `ocr_board.py` | 同上 | 同上 |
| `camera_test_improve.py` (早期版) | `AUTO_EXPOSURE=0.25`(关闭AE) | **最危险 — 写入固件** |

**技术本质**: IMX577 的 UVC 固件会将 `cap.set()` 写入的部分参数持久化到非易失性存储。
设了 brightness=160 后，即使拔插 USB 也无法恢复（实测 mean 仍 > 200）。

#### 修复措施

**1) 清除所有文件的硬件参数**

```python
# 修改前 (危险):
def setup_camera(cap):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)    # ← 导致过曝
    cap.set(cv2.CAP_PROP_EXPOSURE, -3)
    cap.set(cv2.CAP_PROP_GAIN, 100)
    # ...

# 修改后 (安全):
def setup_camera(cap):
    """只设分辨率/MJPG格式, 不动任何进光量参数"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
```

涉及文件: `camera_test.py`, `ocr_workflow_onnx.py`, `ocr_board.py`, `ocr_workflow_onnx copy.py`

**2) 摄像头恢复**: 物理断电（拔USB线等待15秒后重插）使固件参数复位。

#### 教训

> **永远不要通过 OpenCV `cap.set()` 设置 UVC 摄像头的亮度/曝光/增益参数！**
>
> 这些值可能被固件持久化，且不同驱动 (DSHOW/MSMF) 行为不一致。
> 正确做法是让摄像头用出厂默认 AE，图像亮度的调整完全由软件预处理层负责。

---

### 13.2 自适应亮度校正 (`_adaptive_brightness_fix`)

#### 动机

同一场景下拍摄的两张照片质量差异大：

| 图片 | mean | std | 平均置信度 |
|:-----|:----:|:---:|:----------:|
| photo_4.jpg (好) | 174 | 27 | **0.947** |
| photo_6.jpg (差) | 155 | 39 | 0.851 |

photo_6 偏亮区域 OCR 置信度大幅下降（底部 0.41~0.68），需要软件补偿。

#### 方案设计

在原有 `preprocess_image()` 之前插入一个**零开销的自适应层**：

```python
def _adaptive_brightness_fix(img):
    """自适应亮度校正 — 仅在画面偏亮时触发"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    
    # 触发条件: mean>145(偏亮) 或 std<50(动态范围窄)
    needs_fix = mean_val > 145 or std_val < 50
    
    if not needs_fix:
        return img  # 画质正常, 零开销跳过
    
    # Gamma 校正 (gamma > 1 压暗亮区)
    if mean_val > 140:
        bright_score = max((mean_val - 140) / 80, 0)
        low_contrast = max((50 - std_val) / 30, 0)
        severity = max(bright_score, low_contrast * 0.7)
        severity = min(severity, 1.0)
        
        # gamma 范围: 1.3(轻微) ~ 2.2(严重过曝)
        gamma = 1.3 + severity * 0.9
        table = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255)
                          for i in range(256)], dtype='uint8')
        img = cv2.LUT(img, table)
    
    return img
```

#### 调用位置

```python
if ENABLE_PREPROCESS:
    img = _adaptive_brightness_fix(img)   # ★ 新增: 自适应校正
    img = preprocess_image(img)            # 原有预处理管线
```

#### Gamma 公式验证

| 输入亮度 | gamma=0.7 | gamma=1.0 | **gamma=1.5** | **gamma=2.0** |
|:--------:|:--------:|:--------:|:-------------:|:-------------:|
| 200 | 234 ❌更亮 | 200 不变 | **163** ✓压暗 | **157** ✓压暗 |
| 150 | 179 ❌更亮 | 150 不变 | **123** ✓压暗 | **88** ✓很暗 |

**关键发现**: `gamma > 1` 才能压暗，`gamma < 1` 反而提亮！（初次实现时搞反）

#### 实测效果

| 图片 | 原始 mean | 校正后 mean | 原始 std | 校后 std | 触发? |
|:-----|:---------:|:-----------:|:--------:|:-------:|:-----:|
| photo_4 | 173.7 | **130.1** | 26.7 | **43.2** | YES |
| photo_6 | 154.8 | **122.1** | 39.4 | **46.2** | YES |

---

### 13.3 预处理管线调优 — "雷蒙欣"修复

#### 问题现象

`photo_6.jpg` 中 **"商品名称：雷蒙欣"** 被识别为 **"商品名称：需蒙政"**：
- "雷" → "需"
- "欣" → "政"

但右上角 logo 区域的大字 **"雷蒙欣"** 始终正确识别 [0.99]。

#### 根因定位（逐步排查实验）

对 `preprocess_image()` 的三个步骤逐一测试：

| 步骤 | 商品名称结果 | 置信度 | 右上角"雷蒙欣" |
|:-----|:------------|:------:|:--------------:|
| 0. 原图 (无预处理) | **雷蒙欣** ✅ | [0.94] | [0.99] ✅ |
| 1. +锐化 (UnsharpMask) | **雷蒙欣** ✅ | [0.84] | [0.99] ✅ |
| **2. +CLAHE (clipLimit=3.0)** | **雷蒙政** ❌ | [0.86] | [0.99] ✅ |
| **3. +去噪 (h=6)** | **需蒙政** ❌❌ | [0.94] | [0.99] ✅ |

**元凶确认**:
- **CLAHE clipLimit=3.0 过强** → 中小号汉字笔画变形 ("欣" → "政")
- **fastNlMeansDenoising h=6** → 模糊细笔画 ("雷" → "需")

#### 参数网格搜索

| CLAHE clipLimit | 去噪 h | 商品名称 | 判定 |
|:---------------:|:------:|:--------:|:----:|
| 2.0 | 无 | 雷蒙政 ❌ | CLAHE仍太强 |
| **2.5** | **无** | **雷蒙欣 ✅** | **★ 最优** |
| 2.5 | h=4 | 雷蒙政 ❌ | 去噪破坏 |
| 3.0 | 无 | 雷蒙政 ❌ | 当前配置有问题 |
| 3.0 | h=6 (旧配置) | 需蒙政 ❌❌ | 双重破坏 |

#### 最终修改

```python
def preprocess_image(img):
    # Step 1: Unsharp Mask 锐化 — 保持不变
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 1.45, blurred, -0.45, 0)

    # Step 2: CLAHE — ★ clipLimit: 3.0 → 2.5
    #   原因: CL=3.0 时"欣"字笔画变形→被识别为"政"
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # 改这里
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 3: 去噪 — ★ 已移除
    #   原 fastNlMeansDenoisingColored(h=6) 导致"雷→需"、"欣→政"

    return result
```

#### 修复前后对比

| 指标 | 修改前 | 修改后 | 变化 |
|:----:|:------:|:------:|:----:|
| "商品名称："识别 | 需蒙政 [0.87] ❌ | **雷蒙欣** ✅ | 关键字段正确 |
| 右上角"雷蒙欣" | (淹没在低分结果中) | **[0.99]** ✅ | 完美 |
| **平均置信度** | **0.851** | **0.930** | **+7.9%** |
| 总耗时 (photo_6) | 6.92s | 6.51s | **-0.4s** (去掉了去噪) |
| photo_4 置信度 | 0.947 | ~0.95 | 基本持平 |

---

### 13.4 水印过滤增强

#### 新增过滤项

`camera_test.py` 在拍照时会在左上角叠加 OSD 信息（分辨率、FPS 等），被 OCR 误识别：

| OSD 原文 | OCR 识别结果 | 原过滤关键词 | 是否匹配 |
|:---------|:------------|:------------|:-------:|
| `Photos: 5` | `Photas: 5` | `"Photos:"` | ❌ 不匹配! |

**原因**: OCR 把 `Photos` 识别成了 `Photas`（P→Pa 错识别），原过滤规则 `"Photos:"` 无法命中。

```python
FILTER_KEYWORDS = [
    "MJPG", "fps", "CPU:", "RAM:", "App:",
    "Photos:", "Photas:",          # ← 新增 "Photas:"
    "SPACE:", "shot", "quit",
]
```

---

### 13.5 最终预处理管线架构

```
原始图像 (1920×1080)
    │
    ▼
┌─────────────────────────────┐
│ _adaptive_brightness_fix()  │  ★ 新增 (2026-04-29)
│   触发条件: mean>145或std<50 │
│   处理: Gamma 压暗 (γ=1.3~2.2)│
│   不触发时: 零开销直接返回     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ preprocess_image()           │
│                              │
│  Step 1: Unsharp Mask 锐化   │  weight=1.45, sigma=2.0
│         ↓                    │
│  Step 2: CLAHE 对比度增强     │  ★ clipLimit: 3.0→2.5
│         (tileGridSize=8×8)    │
│                              │
│  Step 3: ~~去噪~~            │  ★ 已移除 (破坏中小字号)
│                              │
└──────────────┬──────────────┘
               ▼
        PP-OCRv4 Det → Cls → Rec
               │
               ▼
        结果输出 (+水印过滤)
```

---

### 13.6 经验教训汇总

| # | 教训 | 适用范围 |
|:-:|:-----|:---------|
| 1 | **不要通过 `cap.set()` 设 UVC 摄像头的亮度/曝光/增益** | 所有 USB 摄像头项目 |
| 2 | **Gamma 校正公式**: `output = input^gamma`, **gamma>1 才压暗** | 图像处理 |
| 3 | **CLAHE clipLimit 对 OCR 是敏感参数** — 太强会导致笔画变形 | 文档 OCR 预处理 |
| 4 | **fastNlMeansDenoising 对中小字号有害** — 用锐化+CLAHE 替代去噪 | 低分辨率文字识别 |
| 5 | **逐步排查法定位问题**: 原图→Step1→Step2→Step3 逐一验证 | 调试管线类 Bug |
| 6 | **水印过滤需要考虑 OCR 的错别字变体** (如 Photos→Photas) | 结果后处理 |
| 7 | **摄像头画面抽搐/出现两条明显黑线** → 换 USB 口即可解决，非代码问题 | USB 摄像头硬件排查 |

---

## 十四、药品说明书智能过滤 & 检测漏检根因分析 (2026-04-29)

> **背景**: OCR 识别药品说明书时输出大量元信息（执行标准、批准文号、生产企业等），
> 用户实际只关心成份/适应症/用法用量/不良反应等核心章节。
> 同时发现 photo_12 的注意事项第14条整行被 Det 模型漏检。
> **结论**: 实现可配置的章节级屏蔽开关 + box_thresh 多图 A/B 测试验证保持原参数。

---

### 14.1 药品说明书元信息屏蔽 (`HIDE_DRUG_META`)

#### 需求

OCR 输出的说明书包含大量"不看也没关系"的法定信息：

```
【执行标准】国家药品标准新药转正标准第32册  [0.95]
【批准文号】国药准字H10980262
【说明书修订日期】2020年09月16日
【上市许可持有人】名称：赤峰蒙欣药业有限公司 / 地址：... / 邮编：...
【生产企业】企业名称：... / 生产地址：... / 电话：... / 传真：...
【包装】【有效期】
```

需要一种**按章节标题自动屏蔽**的机制，且能一键切换开/关。

#### 方案设计

```python
# 配置区 (ocr_workflow_onnx.py 第49~60行)
HIDE_DRUG_META = 1   # 0=显示全部  1=屏蔽元信息章节

# 需要屏蔽的章节标题 (以【开头】结尾)
HIDE_SECTION_HEADERS = [
    "【执行标准】", "【批准文号】", "【说明书修订日期】",
    "【上市许可持有人】", "【生产企业】", "【包装】", "【有效期】",
]
```

#### 核心算法 — 状态机式章节追踪

```python
def print_result(result, source_name=""):
    # ... 关键词过滤 (FILTER_KEYWORDS) ...

    # ★ 屏蔽药品说明书元信息章节 — 状态机
    if HIDE_DRUG_META:
        hidden = 0
        hiding = False          # 当前是否处于"隐藏章节"状态
        output = []
        for text, score in filtered:
            is_section_header = any(
                text.strip().startswith(h) or h in text
                for h in HIDE_SECTION_HEADERS
            )
            is_any_section = '【' in text and '】' in text  # 任意章节标题

            if is_section_header:
                hiding = True        # 进入隐藏模式
                hidden += 1
                continue
            elif hiding and is_any_section:
                hiding = False       # 遇到非隐藏的新章节 → 退出隐藏
                output.append((text, score))
            elif not hiding:
                output.append((text, score))   # 正常输出
            else:
                hidden += 1         # 隐藏章节内的行 → 跳过
        filtered = output
```

**设计要点**:

| 要点 | 实现 | 原因 |
|:-----|:-----|:-----|
| 状态机模式 | `hiding` bool 变量 | 比"删除匹配标题后 N 行"更健壮（不同章节数量不同） |
| 退出条件 | 遇到任意 `【xxx】` 新标题 | 不依赖固定行数或位置 |
| 标题匹配策略 | `startswith()` + `in` 双重匹配 | 兼容 `【生产企业】` 和 `【生产企业】名称:` 两种格式 |
| 一键切换 | `HIDE_DRUG_META = 0/1` | 调试时关掉看完整结果，上线时开启 |

#### 实测效果 (photo_12.jpg)

| 指标 | 屏蔽前 (HIDE=0) | 屏蔽后 (HIDE=1) |
|:----:|:---------------:|:---------------:|
| 总检测段数 | 82 | 显示 59 |
| 被屏蔽段数 | — | **23 条元信息** |
| 平均置信度 | 0.967 | 0.967 (不变) |
| 状态栏提示 | — | `识别到82段 | 屏蔽23条元信息 | 显示59段` |

被成功屏蔽的内容: 【执行标准】【批准文号】【说明书修订日期】【上市许可持有人】【生产企业】(含名称/地址/电话/传真/网址) 【包装】【有效期】

保留的核心内容: 药品名称、成份、性状、适应症、规格、用法用量、不良反应、禁忌、注意事项、药物相互作用、药理毒理、贮藏

---

### 14.2 注意事项第14条丢失问题深度诊断

#### 问题现象

photo_12.jpg 的注意事项区域输出：

```
[43] 12.请将本品放在儿童不能接触的地方。
[44] 13.儿童必须在成人监护下使用。
[45] 或药师。  [0.94]              ← 第13条的尾巴（换行断开的）
[46] 【药物相互作用】              ← 直接跳到下一章节！第14条不见了
```

而同一张图的另一张拍摄版本 photo_4.jpg 能完整识别：

```
[47] 12请将本品放在儿童不能接触的地方
[48] 13.儿童必须在成人监护下使用。
[49] 14如正在使用其他药品，使用本品前请咨询医部  [0.91]  ← ★ 有!
[50] 或药师、  [0.89]
```

#### 排查步骤

**Step 1 — 确认不是被 HIDE_DRUG_META 误屏蔽**

关闭所有过滤后查看原始 OCR 输出 → 第14条确实不存在于检测结果中 → **Det 检测阶段就丢了**。

**Step 2 — 对比两图同一区域的图像特征**

对 photo_4 和 photo_12 的第13~14条所在区域 (y坐标 75%~90%) 提取统计指标：

| 指标 | **photo_4** (有第14条) | **photo_12** (丢了) |
|:----:|:----------------------:|:-------------------:|
| 全图平均置信度 | ~0.940 | **0.967** (整体更好!) |
| **区域 mean (亮度)** | **159.6** (偏暗) | 181.6 (偏亮) |
| **区域 std (对比度)** | **36.4** (高对比) | **20.7** (低对比!) |
| min ~ max 范围 | 45 ~ 232 (宽动态) | 57 ~ 222 (窄动态) |

#### 根因结论

> **photo_12 整体拍得更好（均分 0.967 vs 0.940），但恰好第14行所在的页面底部边缘区域过亮**
>
> **mean=182 + std=20.7** → 黑字在近乎纯白的背景上 → 文字与背景的对比度不足
> → DBNet Det 模型的概率图响应值低于 `box_thresh=0.3` 的阈值 → 该文本框被过滤掉

这是一个**反直觉的现象**: **图片整体越清晰，局部过亮区域反而越容易丢字**。不是拍得不好，而是"拍得太均匀太亮"导致的局部对比度塌陷。

#### 解决方案探索 — box_thresh A/B 测试

尝试降低检测阈值找回丢失的文字框：

| box_thresh | photo_8 | photo_10 | photo_12 | 综合评价 |
|:----------:|:-------:|:--------:|:--------:|:---------|
| **0.35 (当前)** | 86条 / **均分0.948** / 低分7 | 81条 / 均分0.955 / 低分7 / **第14✗** | 82条 / **均分0.967** / **低分3** / **第14✗** | ★ 基准 |
| 0.30 (折中) | 90条 / 均分0.943 / 低分9 | 83条 / 均分0.955 / 低分5 / **第14✓** | 84条 / 均分0.964 / 低分4 / **第14✓** | photo_8变差 |
| 0.25 (激进) | 同上 | 同上 | 同上 | 与0.30无差异 |

**最终决策: 保持 `box_thresh = 0.35` 不改**

理由：
- photo_8（带OSD水印的图）对阈值降低特别敏感：低分噪声从 7→9 (+2)，总框从 86→90 (+4)
- photo_10/photo_12 找回第14条的收益无法抵消 photo_8 引入的噪声
- 0.30 和 0.25 效果几乎一样，没有"折中空间"

---

### 14.3 技术亮点总结

| # | 亮点 | 技术价值 |
|:-:|:-----|:---------|
| 1 | **状态机式章节追踪** | 比"删N行"/"正则匹配"更健壮，自适应不同章节长度 |
| 2 | **反直觉根因定位** | 通过区域级 mean/std 分析发现"太亮=低对比度=漏检"，推翻"拍得不好"的直觉 |
| 3 | **多图 A/B 参数测试方法论** | 不是凭感觉调参，而是在 3 张图上量化对比 3 个阈值的 6 项指标后再决策 |
| 4 | **可配置一键切换** | `HIDE_DRUG_META=0/1` 让调试和生产用同一个代码库 |

---

### 14.4 经验教训

| # | 教训 | 适用范围 |
|:-:|:-----|:---------|
| 1 | **Det 漏检 ≠ 图片质量差** — 局部区域过亮(高mean+低std)也会导致DBNet概率响应不足 | DBNet系文本检测器 |
| 2 | **调参必须做多图 A/B 测试** — 单张图上"有效"的参数可能在另一张图上引入噪声 | 所有 OCR 参数调优 |
| 3 | **box_thresh 对不同图片敏感度差异大** — 有OSD水印/纹理复杂的图比干净图文档更容易产生噪声框 | 阈值类超参数 |
| 4 | **章节级过滤优于关键词过滤** — 用关键词过滤元信息会误伤正文中的相同词汇（如"生产企业"出现在正文中） | 结构化文档 OCR |

---

*文档更新时间: 2026-04-29 (药品说明书过滤+漏检分析完结)*
*适用于: 嵌赛答辩材料 / 项目交接 / 技术复盘*
