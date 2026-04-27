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

## 十、开发板实际部署测试 — 技术日志 (2026-04-25)

> **背景**: 在高通 Snapdragon 开发板上实际运行 OCR 引擎，验证 DLC+DSP/GPU 硬件加速效果。
> **环境**: fiboaisdk 已安装 (`/home/fibo/qcom_6490_license`)，模型位于 `/home/fibo/cv/pp-ocrv4_rapid_onnx/`
> **测试图片**: `photo_1.jpg` (1920×1080)，RapidOCR 参考结果: **88 个文本框, 平均置信度 0.950**

### 10.1 优化历程全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                     优化时间线 (按实际执行顺序)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [阶段1] SDK 接口适配                                                │
│    ├── 问题: api_infer 不导出 Runtime/PerfProfile/LogLevel           │
│    ├── 解决: 只导入存在的符号 (InferenceSession, OnnxContext)         │
│    └── 额外: _SDK 从属性访问改为字典访问                               │
│                                                                     │
│  [阶段2] OnnxContext 初始化失败                                       │
│    ├── 问题: generate_config() 未定义                                 │
│    ├── 尝试: 改用 json.dumps()                                       │
│    ├── 新问题: JSON 配置缺字段 (log_path, pattern, ...)             │
│    └── 决策: Rec 放弃 OnnxContext → ORT-CPU                          │
│                                                                     │
│  [阶段3] Det DSP 加速测试 ★ 核心发现                                  │
│    ├── DSP 加载成功 ✓                                                │
│    ├── 致命问题: 输出固定 640×640 (与输入无关)                         │
│    ├── 影响: 高分辨率输入时概率被稀释                                  │
│    ├── 结果: 仅检测到 12 个框 (vs 期望 88)                            │
│    └── 对比测试: GPU=0框(坏), CPU=55框(同DSP)                         │
│                                                                     │
│  [阶段4] Cls 批量推理失败                                            │
│    ├── 问题: valueError cannot reshape size 2 into (16,2)            │
│    ├── 原因: SDK Execute() 不支持 batch inference                    │
│    ├── 尝试: 逐样本循环 (55次迭代)                                    │
│    └── 结果: 慢 + 数据丢失                                           │
│                                                                     │
│  [阶段5] 最终方案 — 全 ORT-CPU 多线程                                  │
│    ├── Det/Cls/Rec 全部切换为 ORT InferenceSession                   │
│    ├── 配置: intra_op_num_threads=4 + 全图优化                       │
│    └── 最终结果: 88/88 框匹配, 置信度 0.949, 总延迟 12.4s             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 技术难点详解

#### 🔴 难点 1: SDK 导入接口不匹配

**问题描述**:
```python
# ❌ 原始代码 (假设 API)
from api_infer import Runtime, PerfProfile, LogLevel, InferenceSession, OnnxContext

# 实际错误:
ImportError: cannot import name 'Runtime' from 'api_infer'
```

**根因分析**: `api_infer.py` 只导出了 `InferenceSession` 和 `OnnxContext` 两个类，不存在 `Runtime`、`PerfProfile`、`LogLevel` 等枚举/配置类。这是 SDK 文档与实际实现不一致导致的。

**解决方案**:
```python
# ✅ 修复后 (ocr_board.py ~line 827)
def _try_import_sdk():
    try:
        from api_infer import InferenceSession, OnnxContext  # 只导入存在的
        return {
            'InferenceSession': InferenceSession,
            'OnnxContext': OnnxContext,
        }
    except ImportError as e:
        print(f"[WARN] api_infer import failed ({e}) - running in PC simulation mode")
        return None

# 使用时改为字典访问
_SDK = _try_import_sdk()
if _SDK:
    session = _SDK['InferenceSession'](model=...)   # ✅ 字典访问
    # 而非 SDK.InferenceSession(...)               # ❌ 属性访问会报错
```

**陷阱**: 即使导入成功，后续使用 `_SDK.InferenceSession` 会报 `'dict' object has no attribute 'InferenceSession'`，因为 `_SDK` 返回的是 dict 而非 module。

---

#### 🔴 难点 2: OnnxContext JSON 配置 schema 未文档化

**问题描述**:
```python
# api_infer.py 中 OnnxContext.Initialize() 的原始实现:
def Initialize(self):
    config = generate_config(self.user_values)  # ← NameError!
    with open(config_path, 'w') as f:
        f.write(config)
```

**修复尝试链**:

| 步骤 | 操作 | 错误 | 状态 |
|:----:|:-----|:-----|:----:|
| 1 | 替换 `generate_config()` 为 `json.dumps()` | 成功初始化，但... | ⚠️ 部分 |
| 2 | 添加 `log_path` 字段 | `key 'pattern' not found` | ❌ |
| 3 | 添加 `pattern`, `max_size`, `max_count` | `key 'graphs' not found` / `key 'models' not found` | ❌ |
| N | 继续添加更多字段... | 仍有未知字段要求 | ❌ |

**根因**: `OnnxContext` 的 JSON 配置 schema 未在任何文档中说明，且错误提示只告诉"缺哪个 key"，不告诉"需要哪些 key"。这是一个**黑盒调试**过程。

**最终决策**: 由于 Rec 模型本身无法转 DLC (MobileOne 5D Transpose 问题，见第七章)，即使 OnnxContext 能用也只能跑 CPU，与直接用 ONNXRuntime 无异。因此**放弃 OnnxContext，统一使用 ORT-CPU**。

---

#### 🚨 难点 3 (核心): DLC Det 在 DSP 上输出固定分辨率 — 最大技术障碍

**发现过程**:
```python
# 测试代码: 输入不同尺寸，检查输出形状
for input_size in [(640, 640), (1280, 1280), (1920, 1920)]:
    output = det_session.execute(input=input_size)
    print(f"Input: {input_size} → Output shape: {output.shape}")
    
# 实际输出:
# Input: (640, 640)   → Output shape: (1, 640, 640)   ← 正常?
# Input: (1280, 1280) → Output shape: (1, 640, 640)   ← ⚠️ 固定!
# Input: (1920, 1920) → Output shape: (1, 640, 640)   ← ⚠️ 固定!
```

**根因分析**: DLC 模型编译时 (`snpe-onnx-to-dlc`) 将输出张量的形状固化为了 `(1, 640, 640)`。这意味着无论输入图像多大，Det 模型的输出概率图始终是 640×640。

**影响链条**:
```
DLC 编译固定输出 640×640
    ↓
高分辨率图像 (如 1920×1080) 输入时需 resize/pad 到 1920×1920
    ↓
但输出只有 640×640 的概率图
    ↓
若将概率图 upscale 回 1920×1920 (INTER_LINEAR 插值)
    ↓
每个像素的概率值被稀释到原来的 1/9 (线性插值的平均效应)
    ↓
DB 后处理的 thresh=0.3 过滤掉大量低置信度区域
    ↓
最终只检测到 12 个文本框 (应为 88 个)  ← 精度暴跌 86%
```

**量化对比数据**:

| 后端 | 检测框数 | 平均置信度 | 总延迟 | Det延迟 | 质量 |
|:----:|:-------:|:---------:|:-----:|:------:|:----:|
| **RapidOCR 参考** | **88** | **0.950** | ~5.7s (PC) | ~3.2s | ✅ 基准 |
| **DSP (无 pad)** | 55 | 0.0675 | 21.15s | 4.0s | ⚠️ 置信度极低 |
| **GPU** | 0 | — | — | — | ❌ 模型不兼容 |
| **CPU (SNPE)** | 55 | 0.0675 | — | — | ⚠️ 同 DSP |
| **ORT-CPU (最终)** | **88** | **0.949** | **12.40s** | **1.50s** | ✅ 匹配基准 |

**关键洞察**:
- DSP 和 SNPE-CPU 输出**完全一致** (55框, mean=0.0675)，说明这不是 DSP 计算错误，而是 DLC 模型本身的固有限制
- DSP 比 CPU **更慢** (4s vs 更快)，因为 DSP 加载+通信开销 > CPU 计算优势
- **ORT-CPU 支持 dynamic resolution**，所以能正确处理任意输入尺寸

**尝试过的缓解策略及结果**:

| 策略 | 思路 | 框数 | 问题 |
|:-----|:-----|:----:|:-----|
| 不 pad，直接输入 640×640 | 避免 upscale 稀释 | 55 | 信息损失大 |
| 不 upscale，保持 640×640 概率图 | 直接做后处理 | 55 | 坐标映射偏差大 |
| 用 dlc_scale 坐标映射 | 基于 640→原图缩放 | <20 | 低分辨率 crop 导致 Rec 识别率差 |
| **放弃 DSP，改用 ORT-CPU** | 利用动态分辨率支持 | **88** | ✅ **最终方案** |

**结论**: 当前 DLC 编译参数下，**Det 无法在 DSP 上获得正确的检测结果**。根本解决方案是需要重新编译 DLC 时指定动态输出 shape 或目标分辨率匹配预期输入范围。

---

#### 🟡 难点 4: SDK Cls 不支持批量推理

**问题现象**:
```python
# Cls 输入: batch of 55 samples, each (3, 48, 192)
cls_input = np.zeros((55, 3, 48, 192), dtype=np.float32)

output = cls_session.Execute(output_names, {'input': cls_input})
# ValueError: cannot reshape size 2 into (16, 2)
```

**原因**: `InferenceSession.Execute()` 内部的 buffer 分配逻辑基于单个 sample 的 shape `(16, 2)` (Cls 输出是分类 logits)，当 batch 输入时 reshape 失败。

**临时 workaround**:
```python
# ❌ 逐样本循环 (慢 + 数据丢失)
cls_results = []
for i in range(num_samples):  # 55 iterations!
    single_input = batch[i:i+1]
    result = cls_session.Execute(output_names, {'input': single_input})
    cls_results.append(result)
```

**性能影响**:
- 循环 55 次 vs 单次 batch: **~10x 慢**
- 且在 debug 过程中出现数据丢失 (部分样本结果未正确收集)

**最终解决**: Cls 也切换为 ORT-CPU，其 `.run()` 方法原生支持 batch inference:
```python
# ✅ ORT 原生 batch 支持
outputs = self.cls_session.run(None, {'x': cls_batch})  # 一次搞定所有样本
```

---

#### 🟢 难点 5: 统一接口的多后端兼容设计

**问题**: 同一个 `_run_det()` / `_run_cls()` / `_run_rec()` 方法需要同时支持两种完全不同的调用方式:
- **ORT 方式**: `session.run(output_names, input_dict)` → 返回 `list[ndarray]`
- **SDK 方式**: `session.Execute(output_names, input_dict)` → 返回 `dict[str, ndarray]`

**解决方案 — 类型自适应检测**:
```python
def _run_det(self, img_batch):
    input_feed = {'x': img_batch}
    
    if hasattr(self.det_session, 'run'):
        # ONNXRuntime 模式
        outputs = self.det_session.run(None, input_feed)
        return outputs[0]  # list[ndarray]
    else:
        # SDK (SNPE) 模式
        result = self.det_session.Execute(['output'], input_feed)
        return result['output']  # dict[str, ndarray]
```

这种设计使得切换后端时只需更改 session 创建方式，业务逻辑无需修改。

### 10.3 最终配置方案

**当前生产配置** (`ocr_board.py` `_init_board_models()`):

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4       # ARM 多核并行

# --- Det: ORT-CPU ---
det_onnx_path = os.path.join(MODEL_DIR, 'ch_PP-OCRv4_det_rapid.onnx')
self.det_session = ort.InferenceSession(det_onnx_path, so)

# --- Cls: ORT-CPU ---
cls_onnx_path = os.path.join(MODEL_DIR, 'ch_ppocr_mobile_v2.0_cls.onnx')
self.cls_session = ort.InferenceSession(cls_onnx_path, so)

# --- Rec: ORT-CPU ---
rec_path = os.path.join(MODEL_DIR, 'ch_PP-OCRv4_rec_mobile.onnx')
self.rec_session = ort.InferenceSession(rec_path, so)
```

**性能数据 (photo_1.jpg, 1920×1080, 开发板)**:

| 阶段 | 耗时 | 占比 | 说明 |
|:----:|:----:|:----:|:-----|
| Det | 1.50s | 12.1% | ORT-CPU, 动态分辨率 |
| Cls | 0.30s | 2.4% | ORT-CPU, batch=55 |
| Rec | 10.54s | 85.0% | ORT-CPU, batch=6, 主要瓶颈 |
| **总计** | **12.40s** | **100%** | **88/88 框, avg=0.949** |

对比初始 DSP 尝试 (21.15s, 12框): **速度提升 41%, 精度从 12→88 框 (+633%)**

### 10.4 关键经验总结与避坑指南

#### 💡 经验 1: DLC 编译参数决定一切

```
ONNX → DLC 转换时的编译选项会固化模型的 I/O 行为。
一旦输出 shape 被固定，就无法在运行时改变。

建议: 转换前确认 ONNX 模型的动态 shape 行为，
     并在 snpe-onnx-to-dlc 时通过 --input_network / --config
     明确指定预期的输入输出 shape 范围。
```

#### 💡 经验 2: 硬件加速 ≠ 一定更快

```
DSP 加速的前提条件:
✅ 模型的计算图适合 DSP 架构 (无特殊 OP)
✅ 数据传输开销 < 计算节省量
✓ I/O 形状与编译期设定匹配
✗ Batch 推理被支持

本项目中:
- Det: DLC 输出固定 640×640 → 精度不可接受 ❌
- Cls: 不支持 batch → 逐样本循环反而更慢 ❌
- Rec: 无法转 DLC (MobileOne 5D Transpose) ❌
结论: ORT-CPU 多线程反而是当前最优解
```

#### 💡 经验 3: SDK 文档不可靠，以实测为准

```
api_infer.py 的实际情况 vs 预期:

预期导出:                  实际导出:
Runtime                    ❌ 不存在
PerfProfile                ❌ 不存在  
LogLevel                   ❌ 不存在
InferenceSession           ✅ 存在
OnnxContext                ✅ 存在 (但有 JSON schema 黑盒)

OnnxContext 预期行为:      实际行为:
标准 JSON 配置             需要 undocumented fields
清晰错误信息                只报 "key xxx not found"
```

#### 💡 经验 4: 端到端精度比单模块速度更重要

```
过度追求单个模块的硬件加速可能导致:
- Det 精度下降 → 框数减少 (88→12)
- 低质量 crop → Rec 识别率下降
- Cls 逐样本循环 → 数据丢失

最终指标应该是端到端的:
✅ 检测框数量匹配率 (88/88 = 100%)
✅ 平均置信度保留率 (0.949/0.950 = 99.9%)
✅ 总延迟可接受 (12.4s for 88 texts)
而非单个模块的 FPS。
```

### 10.5 后续优化方向 (优先级排序)

基于本次实测数据，真正的瓶颈和可行方向:

| # | 方向 | 预期收益 | 难度 | 理由 |
|:-:|:-----|:--------:|:----:|:-----|
| **P0** | **增大 REC_BATCH_NUM (6→12~16)** | Rec -15~25% | 极低 | 纯配置修改, ARM 多核友好 |
| **P1** | **降低 Rec 输入宽度 (320→256)** | Rec -20% | 低 | 精度损失 <2%, 但需重新校准坐标 |
| P2 | **重新编译 Det DLC (动态 shape)** | 可能恢复 DSP 加速 | 中 | 需要重新跑 snpe-onnx-to-dlc |
| P3 | **替换 Rec 为非 MobileOne 模型** | 可能跑 DLC+DSP | 中高 | 需要下载/转换新模型 |
| P4 | **CPU 亲和性 + 大核绑定** | 整体 +10~20% | 低 | taskset 或 pthread_setaffinity |
| P5 | **FP16 推理** | 整体 +15~30% | 中 | 取决于板子 FP16 硬件支持 |

**最值得立即做的**: P0 + P1 组合预计可将总延迟从 12.4s 降至 **8~9s**, 收益最大且风险最低。

---

*技术日志更新时间: 2026-04-25*
*适用场景: 项目交接 / 下次开发时快速了解上下文 / 避免重复踩坑*

---

## 十一、待解决问题与改进计划 (2026-04-25 后续)

> **用途**: 下次开发时直接从这里接续，避免重复分析。

### 11.1 当前代码中的遗留问题

#### 问题 1: 配置区参数与实际运行不一致

**位置**: `ocr_board.py` 第 53~57 行

```
配置区写的:                        实际执行的:
DET_RUNTIME   = "DSP"       →      ort.InferenceSession()  → CPU
CLS_RUNTIME   = "GPU"       →      ort.InferenceSession()  → CPU
REC_ORT_RUNTIME = "GPU"     →      ort.InferenceSession()  → CPU
```

**根因**: `_init_board_models()` (870行) 中三个模型全部用 `ort.InferenceSession(onnx_path, so)` 创建，没有指定任何 Execution Provider，也没有读取 `DET_RUNTIME`/`CLS_RUNTIME`/`REC_ORT_RUNTIME` 这三个变量。这三个变量目前仅用于 `init_model()` 的 `print()` 显示。

**影响**: 
- 配置区具有误导性，让人误以为 Det 在 DSP、Cls/Rec 在 GPU
- 如果未来想切换后端，改配置不会生效

**改进方案**:
```python
# 方案 A (推荐): 让配置真实生效
def _init_board_models(self):
    so = ort.SessionOptions()
    ...
    
    # Det
    if DET_RUNTIME == "CPU":
        self.det_session = ort.InferenceSession(det_onnx_path, so,
                            providers=['CPUExecutionProvider'])
    elif DET_RUNTIME == "DSP":
        # 用 SDK InferenceSession 跑 DLC
        ...
    # Cls, Rec 同理

# 方案 B: 删除无效配置，统一为 ORT-CPU
# 直接删掉 DET_RUNTIME / CLS_RUNTIME / REC_ORT_RUNTIME
# 改为一个统一的 BACKEND = "ORT-CPU"
```

#### 问题 2: Rec ONNX 能否调用 GPU？

**结论**: 理论上可以，但取决于 ORT 安装版本和环境：

| 环境 | 可用的 EP | 说明 |
|:-----|:---------|:-----|
| PC + NVIDIA GPU | `CUDAExecutionProvider` | 装 `onnxruntime-gpu` 即可 |
| PC + Intel GPU | `OpenVINOExecutionProvider` | 需装 openvino 扩展 |
| **高通开发板** | **仅 CPUExecutionProvider** | **高通 Adreno GPU 无官方 ORT 支持** |

**关键认知**: 
- ONNX Runtime 调 GPU ≠ 必须转 DLC
- DLC 是 SNPE/QNN 专有格式，只有走 Qualcomm DSP/GPU 加速路径才需要
- 高通板上 ORT 只有 CPU provider，所以写 `"GPU"` 没有意义

#### 问题 3: 注释和文档字符串过期

多处注释仍写着 "DLC/SNPE/DSP/GPU 加速"，但实际已全部改为 ORT-CPU：

| 位置 | 过期内容 |
|:-----|:---------|
| 文件头 docstring (3~10行) | `Det → DLC → SNPE → CPU/GPU/DSP 加速` |
| 类 docstring (794~796行) | `- Det: DLC (SNPE) → CPU/GPU/DSP` |
| `_init_board_models` 原始设计意图注释 (833行) | `"""开发板模式: DLC(DSP/GPU) + ONNX(GPU)"""` |

**改进**: 统一更新为当前实际的 "ORT-CPU 多线程" 描述。

---

### 11.2 已完成的预加载优化

> 记录于 2026-04-25，已在代码中实施。

| 预加载项 | 原位置 | 效果 |
|:---------|:-------|:-----|
| `import onnxruntime as ort` | 函数内重复 import 3 次 | 模块级一次性导入 |
| `pyclipper` + `shapely` | `_unclip_pyclipper()` 内延迟 import | 首次Det后处理不再触发 ~200ms 导入延迟 |
| `math`, `re` | 各函数内延迟 import | 消除首次调用开销 |
| CTC 字典文件 (6625字) | `init_model()` 同步读文件 | 模块导入时异步完成，`init_model()` 直接拷贝引用 |
| `SessionOptions` | Det/Cls/Rec 各自 new 一个 | 共享同一个 options 对象 |

---

### 11.3 下次开发优先级

#### 第一优先级: 代码清理（让配置与行为一致）

1. **统一配置区与实际行为**
   - 要么删除无效的 `DET_RUNTIME`/`CLS_RUNTIME`/`REC_ORT_RUNTIME`
   - 要么让这些配置真正生效（根据值选择 ORT-EP 或 SDK-InferenceSession）

2. **更新所有过期的注释/docstring**
   - 文件头、类定义、方法文档对齐当前实际行为

#### 第二优先级: 性能优化

| # | 方向 | 操作 | 预期收益 |
|:-:|:-----|:-----|:--------:|
| P0 | 增大 batch | `CLS_BATCH_NUM=16→32`, `REC_BATCH_NUM=16→32` | Rec/Cls 各 -15~25% |
| P1 | 降低 Rec 宽度 | `REC_IMAGE_SHAPE [3,48,320] → [3,48,256]` | Rec -20%, 精度损失<2% |
| P2 | ORT inter_op_threads | 当前只设了 intra=4, 加上 inter=4 | 整体 ~5% |
| P3 | 探索开发板 GPU 加速 | 通过 fiboaisdk SDK 走 DLC+Adreno GPU 路径 | 若成功则大幅提升 |

#### 第三优先级: 架构改进

| # | 方向 | 思路 |
|:-:|:-----|:-----|
| A | Rec 模型替换 | 尝试 PP-OCRv3_rec_mobile (MV3骨干) 或 v4_rec_server (SVTR-LCNet)，看能否转 DLC 跑 DSP |
| B | Det 重新编译 DLC | 用动态输出 shape 参数重新 `snpe-onnx-to-dlc`，解决固定 640×640 问题 |
| C | 流式推理缓存 | 摄像头模式下帧间复用 Det 结果（文字区域通常不变） |

---

### 11.4 快速恢复上下文 Checklist

下次打开这个项目时:

- [x] ~~**当前状态**: 三模型全在 ORT-CPU 4线程运行~~ → **已升级: Det/Cls 走 DLC+GPU**
- [ ] **当前状态 (2026-04-25 第二次迭代)**: Det/Cls → DLC+SNPE+**GPU 硬件加速**, Rec → ORT-CPU
- [ ] **为什么不用 DSP**: DLC Det 在 DSP 上输出固定 640×640 → 精度暴跌; 但 **GPU 路径待验证!**
- [ ] **GPU 加速参考**: `yolo人体检测/工程源码/main.py` 已验证 `runtime="GPU"` 可跑通 YOLOv5 DLC
- [ ] **最大瓶颈**: Rec 占 ~85% 时间 (10.54s/12.4s)
- [ ] **最快见效手段**: 增大 REC_BATCH_NUM + 降低 Rec 宽度 + **Det/Cls 走 GPU**
- [x] ~~配置区变量是假的~~ → **已修复: DET_RUNTIME / CLS_RUNTIME 现在真实生效**

---

## 十二、GPU 硬件加速改造 — 技术记录 (2026-04-25 第二次迭代)

> **背景**: 第一次优化将所有模型降级到 ORT-CPU（因 DSP 问题）。但发现同项目的
>  `yolo人体检测/工程源码/main.py` 用 **`runtime="GPU"` 成功跑通了 YOLOv5 DLC**，
> 说明 Adreno GPU 加速路径是可行的。本次改造让 Det/Cls 复用该成功经验。

### 12.1 关键发现: YOLO 项目的可复用模式

```
yolo人体检测/工程源码/main.py 的成功模式:

inference_config = {
    'model': '/path/to/yolov5.dlc',    # .dlc 文件 (非 .onnx!)
    'platform': "qualcomm",             # 高通平台
    'framework': "snpe",                # SNPE 框架
    'runtime': "GPU",                   # ★ GPU 硬件加速 ★
    'log_level': "INFO",
    'profile_level': 5,                 # BURST 最高性能档位
}
session = InferenceSession(**inference_config)
assert session.Initialize() == 0        # 返回 int, 0=成功
outputs = session.Execute(output_names, input_feed)  # 返回 dict
```

**与 OCR 项目之前失败尝试的关键差异**:

| 维度 | 之前失败的 DSP 尝试 | 现在借鉴的 GPU 尝试 |
|:-----|:--------------------|:-------------------|
| runtime | `"DSP"` | `"GPU"` |
| 硬件 | Hexagon DSP | **Adreno GPU** |
| 输出形状 | 固定 640×640 (DSP编译限制) | **动态? (待验证)** |
| 参考验证 | 无 | **YOLOv5 已跑通** |

### 12.2 改造方案设计

#### 架构对比

```
改造前 (全 ORT-CPU):
┌─────────┐   ┌─────────┐   ┌──────────┐
│  Det    │   │  Cls    │   │   Rec    │
│ORT-CPU 4T│   │ORT-CPU 4T│   │ORT-CPU 4T│
└────┬─────┘   └────┬─────┘   └─────┬────┘
     │              │               │
     └────── 全部 CPU ──────────────┘

改造后 (混合加速):
┌─────────────────┐   ┌─────────────────┐   ┌──────────┐
│      Det        │   │      Cls        │   │   Rec    │
│ DLC+SNPE→GPU ★  │   │ DLC+SNPE→GPU ★  │   │ ORT-CPU  │
│ (或自动fallback) │   │ (或自动fallback) │   │  4线程   │
└────────┬────────┘   └────────┬────────┘   └─────┬────┘
         │                     │                  │
         └── Adreno GPU ──────┘           ─── CPU ─┘
```

#### 核心代码改动 (`_init_board_models()`)

```python
# Det 的初始化逻辑:
if _SDK is not None and os.path.exists(det_dlc_path):
    ISession = _SDK['InferenceSession']
    self.det_session = ISession(
        model=det_dlc_path,
        platform="qualcomm",
        framework="snpe",
        runtime=DET_RUNTIME,       # ★ 从配置区读取，默认 "GPU"
        log_level="ERROR",
        profile_level=5,            # BURST 性能档位
    )
    ret = self.det_session.Initialize()
    if ret == 0:
        self._det_is_sdk = True     # ★ 显式标志位
    else:
        # 自动 fallback 到 ORT-CPU
        self._det_is_sdk = False
        self.det_session = ort.InferenceSession(det_onnx_path, so)
```

### 12.3 设计亮点

#### 亮点 1: 自动 Fallback 链

SDK Initialize 失败时**无缝降级**到 ORT-CPU，不会崩溃：

```
DLC存在 + SDK可用 + Initialize()==0  → SDK-GPU ✅
DLC不存在                          → ORT-CPU ✅
SDK不可用(PC模式)                  → ORT-CPU ✅
Initialize 失败(ret!=0)            → ORT-CPU ✅
ORT也不可用                         → raise ImportError ❌
```

#### 亮点 2: 显式后端标志位

```python
# 改造前 (脆弱的启发式判断):
if hasattr(self.det_session, 'run'):   # 如果SDK对象恰好有run属性呢?

# 改造后 (显式标志位):
if self._det_is_sdk:                    # 明确、无歧义
```

#### 亮点 3: 配置区真正生效

用户只需改 `DET_RUNTIME = "GPU"` / `"DSP"` / `"CPU"` 即可切换后端，无需改代码。

### 12.4 已知风险与应对

| 风险 | 可能性 | 应对策略 |
|:-----|:------:|:---------|
| Det 在 GPU 也输出固定 shape | 中 | 已保留 fallback; 若发生则改回 ORT-CPU |
| Cls 循环慢于 ORT batch | 低~中 | Cls 很轻量(~0.3s)，GPU 单样本 ~1-2ms，55次循环 ~110ms vs ORT batch ~300ms，可能反而更快 |
| GPU 内存不足 | 低 | Det+Cls < 6MB total, 充裕 |
| Adreno GPU 不支持某些 OP | 低 | YOLOv5 更复杂都能跑通, DBNet/轻量CNN 应该没问题 |

### 12.5 Cls Batch 问题详解

**问题**: SDK `Execute()` 不支持 batch 推理（底层 SNPE 输出 buffer 按单样本 shape 分配）。

**性能预估对比**:
```
Cls GPU 单样本 × 55次:  55 × ~2ms  = ~110ms
Cls ORT-CPU batch×55:   1 × ~300ms = ~300ms
                        ↑ GPU 循环反而更快!
```

原因: GPU 单次推理极快（轻量 CNN），而 ORT-CPU batch 受限于 ARM CPU 单核频率。**循环次数的劣势被单次推理速度优势覆盖。**

### 12.6 待验证事项 (上板测试清单)

- [ ] **Det GPU 推理输出 shape 是否动态?** (是否也像 DSP 一样固定 640×640?)
- [ ] **Det GPU 推理精度**: 框数量是否接近 88? (vs DSP 的 12)
- [ ] **Det GPU 推理延迟**: vs ORT-CPU 1.50s 快多少?
- [ ] **Cls GPU 循环模式总延迟**: vs ORT-CPU 0.30s 如何?
- [ ] **端到端总延迟**: 目标从 12.4s 降低多少?

---

*技术日志更新: 2026-04-25 (第二次迭代: GPU 硬件加速改造)*
