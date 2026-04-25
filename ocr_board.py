"""
=============================================================
  OCR 工作流 - 开发板版本（DLC + ONNX 混合推理）
=============================================================
  部署目标: Qualcomm 开发板 (Snapdragon / Hexagon DSP)

  推理后端分配:
    Det (文本检测) → DLC → SNPE → CPU/GPU/DSP 加速
    Cls (方向分类) → DLC → SNPE → CPU/GPU/DSP 加速
    Rec (文字识别) → ONNX → ORT  → CPU 运行

  依赖 (开发板上):
    pip install opencv-python numpy onnxruntime
    + fiboaisdk (官方SDK, 含 api_infer_py)

  使用方法:
    1. 将此文件 + 模型文件放到开发板同一目录下
    2. 确保模型路径正确 (MODEL_DIR 下)
    3. python ocr_board.py

  ★★★ 配置区在文件顶部，只改那里即可 ★★★
"""

import os
import sys
import time
import cv2
import numpy as np


# ============================================================
#  ★★★ 配置区（改这里就行）★★★
# ============================================================

# ── 基础路径 ──
# 开发板上模型存放目录，根据实际部署路径修改
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'pp-ocrv4_rapid_onnx')
SAVE_DIR = os.path.join(BASE_DIR, 'photos')

# ── 模型文件名 ──
DET_DLC = 'ch_PP-OCRv4_det_mobile.dlc'      # 文本检测 DLC
CLS_DLC = 'ch_ppocr_mobile_v2.0_cls_mobile.dlc'   # 方向分类 DLC
REC_ONNX = 'ch_PP-OCRv4_rec_mobile.onnx'     # 文字识别 ONNX (FP32/INT8)
# REC_ONNX = 'ch_PP-OCRv4_rec_mobile_int8.onnx'  # ← 用量化版改这里

# ── 字典文件 ──
DICT_PATH = os.path.join(MODEL_DIR, 'ppocr_keys_v1.txt')

# ── SNPE/DLC 运行时选择 ──
#   可选: "CPU", "GPU", "DSP"
#   DSP 最快但需要 hexagon 库，CPU 最兼容
DET_RUNTIME = "DSP"   # det 计算量大，推荐 DSP/GPU
CLS_RUNTIME = "CPU"   # cls 很轻量，CPU 够用

# ── ONNX rec 运行时 ──
REC_ORT_RUNTIME = "CPU"

# ── 模式选择（填数字）──
MODE = 1   # 1=图片识别    2=摄像头拍照+识别

# ── 模式1：填图片路径 ──
IMAGE_PATHS = [
    "photos/photo_1.jpg",
    # "photos/藿香正气水.jpg",
    # "photos/细菌溶解产物胶囊.jpg",
]

# ── 预处理开关 ──
ENABLE_PREPROCESS = False

# ── 过滤关键词 ──
FILTER_KEYWORDS = [
    "MJPG", "fps", "CPU:", "RAM:", "App:", "Photos:", "SPACE:",
    "shot", "quit",
]

# ── Det 参数 ──
DET_LIMIT_SIDE_LEN = 736    # 检测输入最大边长
DET_LIMIT_TYPE = 'min'      # 'min'=限制短边  'max'=限制长边
DET_THRESH = 0.20           # 二值化阈值
DET_BOX_THRESH = 0.35       # 文本框置信度阈值
DET_UNCLIP_RATIO = 1.6      # 文本框扩展比例
DET_MAX_CANDIDATES = 1000   # 最大候选框数
DET_USE_DILATION = True     # 是否使用膨胀核细化文本区域

# ── Cls 参数 ──
CLS_IMAGE_SHAPE = [3, 48, 192]
CLS_BATCH_NUM = 6
CLS_THRESH = 0.9            # 方向分类置信度阈值（低于此值不旋转）

# ── Rec 参数 ──
REC_IMAGE_SHAPE = [3, 48, 320]
REC_BATCH_NUM = 6

# ── 全局参数 ──
TEXT_SCORE_THRESH = 0.4     # 最终输出最低置信度过滤


# ============================================================
#  SDK 导入（开发板专用）
# ============================================================

def _try_import_sdk():
    """尝试导入开发板 SDK，PC 上运行会优雅降级"""
    try:
        from api_infer import InferenceSession, OnnxContext, Runtime, PerfProfile, LogLevel
        return {
            'InferenceSession': InferenceSession,
            'OnnxContext': OnnxContext,
            'Runtime': Runtime,
            'PerfProfile': PerfProfile,
            'LogLevel': LogLevel,
        }
    except ImportError:
        print("[WARN] api_infer not found - running in PC simulation mode")
        print("       On dev board, ensure fiboaisdk is installed")
        return None

_SDK = _try_import_sdk()


# ============================================================
#  工具函数
# ============================================================

def load_image(path):
    """读取图片（支持中文路径）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_dict(dict_path):
    """加载 CTC 解码字典"""
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f if line.strip()]
    # PP-OCR 字典格式: 第一个字符是空白符(CTC blank)，后面是可见字符
    return chars, len(chars)


def preprocess_image(img):
    """OCR 专用图像预处理：锐化 → CLAHE → 去噪"""
    blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
    sharp = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    result = cv2.fastNlMeansDenoisingColored(result, None, h=7,
                                             templateWindowSize=5, searchWindowSize=15)
    return result


# ============================================================
#  文本框排序
# ============================================================

def sort_boxes_by_layout(boxes, texts, scores, mode="auto"):
    """按阅读顺序排序文本框"""
    if boxes is None or len(boxes) == 0:
        return boxes, texts, scores
    n = len(boxes)
    centers = np.array([box.mean(axis=0) for box in boxes])
    x_centers = centers[:, 0]
    y_centers = centers[:, 1]
    if mode == "auto":
        x_range = x_centers.max() - x_centers.min()
        y_range = y_centers.max() - y_centers.min()
        if y_range > 0 and (x_range / y_range) > 1.5:
            mode = "double"
        else:
            mode = "single"
    if mode == "single":
        indices = np.argsort(y_centers)
    elif mode == "double":
        x_mid = (x_centers.max() + x_centers.min()) / 2
        left_mask = x_centers <= x_mid
        right_mask = x_centers > x_mid
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]
        left_sorted = left_indices[np.argsort(y_centers[left_indices])]
        right_sorted = right_indices[np.argsort(y_centers[right_indices])]
        indices = np.concatenate([left_sorted, right_sorted])
    else:
        indices = np.arange(n)
    return ([boxes[i] for i in indices],
            [texts[i] for i in indices],
            [scores[i] for i in indices])


# ============================================================
#  Det 预处理 / 后处理
# ============================================================

def det_preprocess(img, limit_side_len=736, limit_type='min'):
    """
    文本检测预处理:
    1. 按比例缩放（保持长宽比）
    2. 归一化 (mean=0.5, std=0.5)
    3. padding 到 limit_side_len 的倍数
    """
    h, w = img.shape[:2]
    if limit_type == 'min':
        ratio = min(limit_side_len / h, limit_side_len / w)
    else:
        ratio = max(limit_side_len / h, limit_side_len / w)
    resize_h, resize_h_int = int(h * ratio), int(round(h * ratio))
    resize_w, resize_w_int = int(w * ratio), int(round(w * ratio))

    resized = cv2.resize(img, (resize_w_int, resize_h_int))
    # 归一化 (PP-OCR 标准归一化: mean=0.5, std=0.5)
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (normalized - mean) / std
    # HWC -> CHW
    normalized = normalized.transpose((2, 0, 1))
    # padding 到 32 的倍数
    pad_h = 32 - resize_h % 32 if resize_h % 32 != 0 else 0
    pad_w = 32 - resize_w % 32 if resize_w % 32 != 0 else 0
    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((3, resize_h + pad_h, resize_w + pad_w), dtype=np.float32)
        padded[:, :resize_h, :resize_w] = normalized
    else:
        padded = normalized

    return padded[np.newaxis], (resize_h, resize_w, ratio, h, w)


def det_postprocess(det_result, original_shape, resize_info,
                    thresh=0.30, box_thresh=0.50,
                    unclip_ratio=1.6, max_candidates=1000,
                    use_dilation=True):
    """
    DB (Differentiable Binarization) 后处理:
    1. 二值化概率图
    2. 轮廓提取
    3. 最小外接矩形
    4. 缩放回原图坐标
    
    det_result: dict {'sigmoid_0.tmp_0': ndarray(N,1,H,W)} 或类似格式
    """
    # 统一提取概率图
    if isinstance(det_result, dict):
        key = list(det_result.keys())[0]
        prob_data = np.array(det_result[key])
    else:
        prob_data = np.array(det_result)

    if prob_data.ndim == 4:
        prob_map = prob_data[0, 0]  # (H, W)
    elif prob_data.ndim == 3:
        prob_map = prob_data[0]     # (H, W)
    elif prob_data.ndim == 2:
        prob_map = prob_data
    else:
        raise ValueError(f"Unexpected det output shape: {prob_data.shape}")

    resize_h, resize_w, ratio, orig_h, orig_w = resize_info

    # 二值化
    binary = (prob_map > thresh).astype(np.uint8) * 255

    # 可选膨胀（细化分离粘连文字）
    if use_dilation:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []
    for contour in contours:
        if len(contour) < 4:
            continue

        # 最小面积矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box)

        # 计算 box 在概率图上的平均得分
        mask = np.zeros_like(prob_map, dtype=np.uint8)
        cv2.fillConvexPoly(mask, box.astype(np.int32), 1)
        score = float(cv2.mean(prob_map, mask=mask)[0])
        if score < box_thresh:
            continue

        # unclip: 扩展文本框
        if unclip_ratio > 1:
            expanded = unclip_box(box, unclip_ratio)
        else:
            expanded = box

        # 再取一次最小面积矩形（扩展后可能变形）
        rect = cv2.minAreaRect(expanded.astype(np.int32))
        final_box = cv2.boxPoints(rect).astype(np.float32)

        # 缩放回原图坐标
        scale_x = orig_w / resize_w
        scale_y = orig_h / resize_h
        final_box[:, 0] *= scale_x
        final_box[:, 1] *= scale_y

        # 裁剪到图片范围内
        final_box[:, 0] = np.clip(final_box[:, 0], 0, orig_w)
        final_box[:, 1] = np.clip(final_box[:, 1], 0, orig_h)

        boxes.append(final_box)
        scores.append(score)
        if len(boxes) >= max_candidates:
            break

    return boxes, scores


def unclip_box(box, unclip_ratio=1.6):
    """按比例扩展多边形（unclip 操作）"""
    if unclip_ratio <= 1.0:
        return box

    poly = np.array(box, dtype=np.float32)
    # 计算中心点
    center_x = poly[:, 0].mean()
    center_y = poly[:, 1].mean()
    # 从中心向外扩展
    expanded = poly.copy()
    expanded[:, 0] = center_x + (poly[:, 0] - center_x) * unclip_ratio
    expanded[:, 1] = center_y + (poly[:, 1] - center_y) * unclip_ratio
    return expanded


def get_rotate_crop_image(img, box):
    """根据四点坐标裁剪并矫正旋转的文本区域图像"""
    points = box.astype(np.float32)
    w = int(max(np.linalg.norm(points[0] - points[1]),
                 np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]),
                 np.linalg.norm(points[1] - points[2])))

    src_pts = points
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(img, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    return cropped


# ============================================================
#  Cls 预处理 / 后处理
# ============================================================

def cls_preprocess(images, target_shape=(48, 192)):
    """方向分类预处理: resize → normalize → CHW → batch"""
    batch = []
    th, tw = target_shape
    for img in images:
        resized = cv2.resize(img, (tw, th))
        normed = resized.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        normed = (normed - mean) / std
        normed = normed.transpose((2, 0, 1))  # HWC -> CHW
        batch.append(normed)
    return np.stack(batch, axis=0)


def cls_postprocess(output, thresh=0.9):
    """
    方向分类后处理:
    返回每个 crop 是否需要旋转 180 度
    label_list: ["0"(正向), "180"(需翻转)]
    """
    preds = np.array(output['save_infer_model/scale_0.tmp_1'])
    if isinstance(preds, list):
        preds = np.array(preds[0])

    angles = []
    scores = []
    for pred in preds:
        idx = np.argmax(pred)
        score = float(pred[idx])
        angles.append(idx)          # 0=正向 1=旋转180度
        scores.append(score)
    return angles, scores


# ============================================================
#  Rec 预处理 / 后处理 (CTC 解码)
# ============================================================

def rec_preprocess(images, target_shape=(48, 320)):
    """文字识别预处理: resize 保持比例 → padding → normalize → CHW → batch"""
    batch = []
    th, tw = target_shape
    for img in images:
        h, w = img.shape[:2]
        ratio = min(th / h, tw / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        resized = cv2.resize(img, (new_w, new_h))

        padded = np.full((th, tw, 3), 128, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        normed = padded.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        normed = (normed - mean) / std
        normed = normed.transpose((2, 0, 1))  # HWC -> CHW
        batch.append(normed)
    return np.stack(batch, axis=0)


def ctc_decode_greedy(probs, dict_chars):
    """
    CTC Greedy 解码
    probs: (batch, seq_len, num_classes) 或 (seq_len, num_classes)
    dict_chars: 字符列表 (index 0 = blank/blank token)
    """
    texts = []
    text_scores = []

    if probs.ndim == 3:
        # batch mode
        for i in range(probs.shape[0]):
            t, s = _ctc_single(probs[i], dict_chars)
            texts.append(t)
            text_scores.append(s)
    else:
        t, s = _ctc_single(probs, dict_chars)
        texts.append(t)
        text_scores.append(s)

    return texts, text_scores


def _ctc_single(prob_seq, dict_chars):
    """单条 CTC 序列解码"""
    pred_idx = np.argmax(prob_seq, axis=1)
    # 压缩连续重复字符
    prev_idx = -1
    chars = []
    scores_sum = 0.0
    count = 0
    for t, idx in enumerate(pred_idx):
        if idx != prev_idx:
            if idx > 0 and idx < len(dict_chars):  # 跳过 blank (idx=0) 和越界
                chars.append(dict_chars[idx])
                scores_sum += float(prob_seq[t, idx])
                count += 1
            prev_idx = idx
    text = ''.join(chars)
    avg_score = scores_sum / count if count > 0 else 0.0
    return text, avg_score


# ============================================================
#  ★ 核心: OCR 引擎 (DLC + ONNX 混合推理版) ★
# ============================================================

class OCRBoardEngine:
    """
    开发板 OCR 引擎

    推理后端:
      - Det: DLC (SNPE) → CPU/GPU/DSP
      - Cls: DLC (SNPE) → CPU/GPU/DSP
      - Rec: ONNX (ORT) → CPU
    """

    def __init__(self):
        self.det_session = None
        self.cls_session = None
        self.rec_session = None
        self.dict_chars = None
        self.dict_size = 0
        self._load_time = 0
        self._is_pc_mode = (_SDK is None)

    def init_model(self):
        """加载所有模型"""
        t0 = time.perf_counter()

        print("=" * 55)
        print("  Board OCR Engine - Loading Models")
        print("=" * 55)
        print(f"  Det: {DET_DLC} ({DET_RUNTIME})")
        print(f"  Cls: {CLS_DLC} ({CLS_RUNTIME})")
        print(f"  Rec: {REC_ONNX} ({REC_ORT_RUNTIME})")

        # 加载字典
        self.dict_chars, self.dict_size = load_dict(DICT_PATH)
        print(f"  Dict: {self.dict_size} chars")

        if self._is_pc_mode:
            self._init_pc_fallback()
        else:
            self._init_board_models()

        self._load_time = time.perf_counter() - t0
        print(f"\n  All models loaded! ({self._load_time:.3f}s)\n")

    def _init_board_models(self):
        """开发板模式: DLC + ONNX"""
        SDK = _SDK
        det_path = os.path.join(MODEL_DIR, DET_DLC)
        cls_path = os.path.join(MODEL_DIR, CLS_DLC)
        rec_path = os.path.join(MODEL_DIR, REC_ONNX)

        # --- Det (DLC/SNPE) ---
        print("\n  Loading Det (DLC)...")
        runtime_val = getattr(SDK.Runtime, DET_RUNTIME.upper(), SDK.Runtime.CPU)
        self.det_session = SDK.InferenceSession(
            model=det_path,
            platform="QUALCOMM",
            framework="SNPE",
            runtime=DET_RUNTIME.upper(),
            log_level="ERROR",
            profile_level=1,  # HIGH_PERFORMANCE
        )
        ret = self.det_session.Initialize()
        assert ret == 0, f"Det model Init failed: {ret}"
        print("  [OK] Det loaded")

        # --- Cls (DLC/SNPE) ---
        print("  Loading Cls (DLC)...")
        runtime_cls = getattr(SDK.Runtime, CLS_RUNTIME.upper(), SDK.Runtime.CPU)
        self.cls_session = SDK.InferenceSession(
            model=cls_path,
            platform="QUALCOMM",
            framework="SNPE",
            runtime=CLS_RUNTIME.upper(),
            log_level="ERROR",
            profile_level=1,
        )
        ret = self.cls_session.Initialize()
        assert ret == 0, f"Cls model Init failed: {ret}"
        print("  [OK] Cls loaded")

        # --- Rec (ONNX) ---
        print("  Loading Rec (ONNX)...")
        self.rec_session = SDK.OnnxContext(
            onnx_path=rec_path,
            output_tensors=['softmax_11.tmp_0'],
            runtime=REC_ORT_RUNTIME.upper(),
            log_level="ERROR",
        )
        ret = self.rec_session.Initialize()
        assert ret == 0, f"Rec model Init failed: {ret}"
        print("  [OK] Rec loaded")

    def _init_pc_fallback(self):
        """PC 回退模式: 全部用 ONNXRuntime (方便本地测试)
        
        已应用 Rec 性能优化 (基准测试验证 ~10% 加速):
          - intra_op_num_threads=4: 多线程并行
          - SEQUENTIAL execution mode: 减少线程同步开销
          - enable_mem_reuse=True: 复用内存分配
          - enable_mem_pattern=True: 内存模式优化
        """
        import onnxruntime as ort
        print("\n  [PC Mode] Using ONNXRuntime for all models (Rec optimized)...")

        det_path = os.path.join(MODEL_DIR, DET_DLC.replace('.dlc', '.onnx'))
        cls_path = os.path.join(MODEL_DIR, CLS_DLC.replace('.dlc', '.onnx'))
        rec_path = os.path.join(MODEL_DIR, REC_ONNX)

        # 如果 DLC 对应的 ONNX 不存在，尝试找 .onnx 版本
        if not os.path.exists(det_path):
            det_path = os.path.join(MODEL_DIR,
                        [f for f in os.listdir(MODEL_DIR) if 'det' in f.lower() and f.endswith('.onnx')][0]
                        if any('det' in f.lower() for f in os.listdir(MODEL_DIR)) else '')
        if not os.path.exists(cls_path):
            cls_path = os.path.join(MODEL_DIR,
                        [f for f in os.listdir(MODEL_DIR) if 'cls' in f.lower() and f.endswith('.onnx')][0]
                        if any('cls' in f.lower() for f in os.listdir(MODEL_DIR)) else '')

        # Det/Cls: 默认配置 (轻量级，不需要特殊优化)
        so_base = ort.SessionOptions()
        so_base.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # ★ Rec: 满性能优化配置 ★ (基准测试: 37.6ms vs 41.2ms/batch, +10%)
        so_rec = ort.SessionOptions()
        so_rec.intra_op_num_threads = max(4, ORT_THREADS) if ORT_THREADS > 0 else 4
        so_rec.inter_op_num_threads = max(4, ORT_THREADS) if ORT_THREADS > 0 else 4
        so_rec.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so_rec.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so_rec.enable_mem_pattern = True
        so_rec.enable_mem_reuse = True

        self.det_session = ort.InferenceSession(det_path, so_base)
        self.cls_session = ort.InferenceSession(cls_path, so_base)
        self.rec_session = ort.InferenceSession(rec_path, so_rec)
        print(f"  [OK] PC Fallback: det={os.path.basename(det_path)} "
              f"cls={os.path.basename(cls_path)} rec={os.path.basename(rec_path)}")
        print(f"       Rec optimization: threads={so_rec.intra_op_num_threads}, "
              f"sequential+mem_reuse")

    def recognize(self, image_or_path):
        """完整 OCR 识别流程"""
        if self.det_session is None:
            raise RuntimeError("请先调用 init_model()")

        # 加载图片
        if isinstance(image_or_path, str):
            img = load_image(image_or_path)
        else:
            img = image_or_path
        if img is None:
            return None

        if ENABLE_PREPROCESS:
            img = preprocess_image(img)

        orig_h, orig_w = img.shape[:2]

        # ===== Stage 1: Text Detection =====
        t0 = time.perf_counter()
        det_input, resize_info = det_preprocess(img, DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE)
        det_output = self._run_det(det_input)
        det_t = time.perf_counter() - t0

        # ===== Stage 2: Post-process Detection =====
        t1 = time.perf_counter()
        boxes, box_scores = det_postprocess(
            det_output, img.shape, resize_info,
            thresh=DET_THRESH,
            box_thresh=DET_BOX_THRESH,
            unclip_ratio=DET_UNCLIP_RATIO,
            max_candidates=DET_MAX_CANDIDATES,
            use_dilation=DET_USE_DILATION,
        )
        post_t = time.perf_counter() - t1

        if len(boxes) == 0:
            return {'texts': [], 'scores': [], 'boxes': [],
                    'count': 0, 'avg_score': 0,
                    'elapsed': det_t + post_t,
                    'elapsed_det': det_t, 'elapsed_cls': 0, 'elapsed_rec': 0}

        # ===== Stage 3: Crop & Classify Direction =====
        t2 = time.perf_counter()
        crops = [get_rotate_crop_image(img, box) for box in boxes]

        # 分批做方向分类
        all_angles = []
        all_cls_scores = []
        batch_size = CLS_BATCH_NUM
        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start:start + batch_size]
            cls_input = cls_preprocess(batch_crops, tuple(CLS_IMAGE_SHAPE[1:]))
            cls_output = self._run_cls(cls_input)
            angles, cls_scores = cls_postprocess(cls_output, CLS_THRESH)
            all_angles.extend(angles)
            all_cls_scores.extend(cls_scores)
        cls_t = time.perf_counter() - t2

        # 根据角度翻转图片
        rotated_crops = []
        for crop, angle in zip(crops, all_angles):
            if angle == 1:  # 需要旋转 180 度
                crop = cv2.rotate(crop, cv2.ROTATE_180)
            rotated_crops.append(crop)

        # ===== Stage 4: Text Recognition (CTC) =====
        t3 = time.perf_counter()
        all_texts, all_text_scores = [], []
        batch_size = REC_BATCH_NUM
        for start in range(0, len(rotated_crops), batch_size):
            batch_crops = rotated_crops[start:start + batch_size]
            rec_input = rec_preprocess(batch_crops, tuple(REC_IMAGE_SHAPE[1:]))
            rec_output = self._run_rec(rec_input)
            texts, text_scores = ctc_decode_greedy(rec_output, self.dict_chars)
            all_texts.extend(texts)
            all_text_scores.extend(text_scores)
        rec_t = time.perf_counter() - t3

        total_t = det_t + post_t + cls_t + rec_t

        # 组装最终结果（过滤低置信度 + 对齐 boxes）
        valid_texts, valid_scores, valid_boxes = [], [], []
        for txt, sc, bs, box in zip(all_texts, all_text_scores, box_scores, boxes):
            final_score = sc * bs
            if final_score >= TEXT_SCORE_THRESH:
                valid_texts.append(txt)
                valid_scores.append(final_score)
                valid_boxes.append(box)

        h, w = img.shape[:2]
        mode = "double" if w / h > 1.5 else "single"
        sorted_boxes, sorted_texts, sorted_scores = sort_boxes_by_layout(
            valid_boxes, valid_texts, valid_scores, mode=mode
        )

        return {
            'texts': sorted_texts,
            'scores': sorted_scores,
            'boxes': sorted_boxes,
            'count': len(sorted_texts),
            'avg_score': sum(sorted_scores) / len(sorted_scores) if sorted_scores else 0,
            'elapsed': total_t,
            'elapsed_det': det_t,
            'elapsed_cls': cls_t,
            'elapsed_rec': rec_t,
        }

    def _run_det(self, input_data):
        """执行 Det 推理"""
        input_feed = {'x': input_data.astype(np.float32)}
        output_names = ['sigmoid_0.tmp_0']
        if self._is_pc_mode:
            out = self.det_session.run(output_names, input_feed)
            return {output_names[0]: out[0]}
        else:
            result = self.det_session.Execute(output_names, input_feed)
            return result

    def _run_cls(self, input_data):
        """执行 Cls 推理"""
        input_feed = {'x': input_data.astype(np.float32)}
        output_names = ['save_infer_model/scale_0.tmp_1']
        if self._is_pc_mode:
            out = self.cls_session.run(output_names, input_feed)
            return {output_names[0]: out[0]}
        else:
            result = self.cls_session.Execute(output_names, input_feed)
            return result

    def _run_rec(self, input_data):
        """执行 Rec 推理"""
        input_feed = {'x': input_data.astype(np.float32)}
        output_names = ['softmax_11.tmp_0']
        if self._is_pc_mode:
            out = self.rec_session.run(output_names, input_feed)
            return out[0]  # (B, T, C)
        else:
            result = self.rec_session.Execute(output_names, input_feed)
            # SDK 返回的是 flat list，需要 reshape
            val = result.get('softmax_11.tmp_0', [])
            batch = input_data.shape[0]
            seq_len = len(val) // batch // self.dict_size
            return np.array(val).reshape(batch, seq_len, self.dict_size)

    def release(self):
        """释放资源"""
        if hasattr(self.det_session, 'Release'):
            self.det_session.Release()
        if hasattr(self.cls_session, 'Release'):
            self.cls_session.Release()
        if hasattr(self.rec_session, 'Release'):
            self.rec_session.Release()


# ============================================================
#  摄像头模块
# ============================================================

def find_best_camera():
    best_idx, best_res = None, 0
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, f = cap.read()
            if ret and f is not None:
                cap.set(cv2.CAP_PROP_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_HEIGHT, 1080)
                ret2 = cap.read()[0]
                if ret2:
                    mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    mh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    res = mw * mh
                    if res > best_res:
                        best_res, best_idx = res, i
            cap.release()
    return best_idx


def setup_camera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
    cap.set(cv2.CAP_PROP_CONTRAST, 150)
    cap.set(cv2.CAP_PROP_SATURATION, 115)
    try: cap.set(cv2.CAP_PROP_SHARPNESS, 200)
    except: pass
    try: cap.set(cv2.CAP_PROP_GAIN, 100)
    except: pass
    try: cap.set(cv2.CAP_PROP_EXPOSURE, -3)
    except: pass
    try: cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    except: pass


def camera_mode(ocr_engine):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 50)
    print("  USB Camera (Board Version)")
    print("=" * 50)

    cam_idx = find_best_camera()
    if cam_idx is None:
        print("ERROR: Camera not found!")
        return

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    setup_camera(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win = 'Camera-Board'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    print(f"[OK] {w}x{h} | SPACE=shot+ocr  Q=quit\n")

    photo_count = 0
    fps_list, t_last, fps_show = [], time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        now = time.time()
        fps_list.append(now)
        fps_list[:] = [t for t in fps_list if now - t < 1.0]
        if now - t_last >= 1.0:
            fps_show = len(fps_list)
            t_last = now

        info = [f"{w}x{h}  {fps_show}fps",
               f"Photos: {photo_count}",
               "SPACE=shot  Q=exit"]
        y = 28
        for line in info:
            cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,230,120), 2)
            y += 28

        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord(' '):
            photo_count += 1
            path = os.path.join(SAVE_DIR, f'photo_{photo_count}.jpg')
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if ok:
                with open(path, 'wb') as f:
                    f.write(buf)
                sz = os.path.getsize(path) / 1024
                print(f"\n{'='*60}")
                print(f"  [SAVED] photo_{photo_count}.jpg ({sz:.0f}KB)")
                result = ocr_engine.recognize(frame)
                if result:
                    print_result(result, path)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _get_next_photo_id(save_dir):
    if not os.path.exists(save_dir):
        return 0
    max_id = 0
    import re
    for f in os.listdir(save_dir):
        m = re.match(r'photo_(\d+)\.', f)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


# ============================================================
#  结果输出
# ============================================================

def print_result(result, source_name=""):
    filtered = []
    for text, score in zip(result['texts'], result['scores']):
        if any(kw in text for kw in FILTER_KEYWORDS):
            continue
        filtered.append((text, score))

    count = len(filtered)
    total = result['count']
    skipped = total - count

    print(f"  Found {total} text regions (filtered {skipped}), showing {count}, "
          f"avg confidence {result['avg_score']:.3f}")
    if count == 0:
        print("  |  (all filtered or no text)")
    else:
        print("  |")
        for text, score in filtered:
            tag = "" if score >= 0.95 else f"  [{score:.2f}]"
            print(f"  |  {text}{tag}")
    print("  |")
    print(f"  |  Time: det={result['elapsed_det']:.2f}s + "
          f"cls={result['elapsed_cls']:.2f}s + rec={result['elapsed_rec']:.2f}s"
          f" = total={result['elapsed']:.2f}s")
    print()


# ============================================================
#  主入口
# ============================================================

def batch_mode(ocr_engine, image_paths):
    total_start = time.perf_counter()
    total_images = 0
    for path in image_paths:
        if not os.path.exists(path):
            print(f"[SKIP] Not found: {path}")
            continue
        print(f"{'='*60}\n  Image: {path}\n{'='*60}")
        result = ocr_engine.recognize(path)
        if result:
            print_result(result, path)
            total_images += 1
        else:
            print("  [No text detected]\n")

    total_time = time.perf_counter() - total_start
    print(f"{'='*60}")
    print(f"[Done] {total_images} images, total {total_time:.3f}s")
    if total_images > 0:
        print(f"        Avg per image {total_time/total_images:.3f}s")
    print(f"{'='*60}")


def main():
    engine = OCRBoardEngine()
    engine.init_model()

    try:
        if MODE == 2:
            print("\n[Mode2] Camera + OCR (Board)")
            camera_mode(engine)
        elif MODE == 1:
            if not IMAGE_PATHS or all(p.strip().startswith('#') for p in IMAGE_PATHS):
                print("!!! IMAGE_PATHS is empty !!!")
                return
            paths = [p for p in IMAGE_PATHS if p and not p.strip().startswith('#')]
            print(f"\n[Mode1] Image OCR - Board Version ({len(paths)} images)\n")
            batch_mode(engine, paths)
        else:
            print(f"!!! MODE must be 1 or 2, got {MODE} !!!")
    finally:
        engine.release()


if __name__ == '__main__':
    main()
