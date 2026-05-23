"""
=============================================================
  OCR 工作流 - V3 改进版（基于 ocr_board.py 最优配置）
=============================================================
  改进点 (vs copy 2.py):
    1. ★ 强制 Cls 用 ORT-CPU (DLC-GPU 方向分类误旋转导致19%文本丢失)
    2. ★ 强制 Det 用 ORT-CPU (DLC 固定640输出限制检测精度)
    3. ★ 修复 _run_rec SDK路径的 dict_size bug (6622→6625)
    4. ★ 支持 INT8 Rec 模型选项 (REC_ONNX 一行切换)
    5. ★ 移除 DLC 自动探测逻辑 (避免意外激活不可靠后端)

  推理后端:
    Det → ONNX → ORT-CPU (唯一正确方案, 88框全检出)
    Cls → ONNX → ORT-CPU (准确方向分类, 无误旋转)
    Rec → ONNX → ORT-CPU (FP32 或 INT8, 一行切换)

  使用方法:
    python ocr_board_v3.py
"""

import os
import sys
import time
import re
import math
import cv2
import numpy as np

# ★ 预加载重型依赖 — 避免推理时首次调用的 import 开销 ★
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import pyclipper
    from shapely.geometry import Polygon as ShapelyPolygon
    _has_pyclipper = True
except ImportError:
    _has_pyclipper = False

# ★ 预加载 CTC 字典 — init_model() 时直接可用，无需等待文件 I/O ★
_PRELOAD_DICT = None
_PRELOAD_DICT_SIZE = 0
try:
    _DICT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'pp-ocrv4_rapid_onnx', 'ppocr_keys_v1.txt'
    )
    if os.path.exists(_DICT_PATH):
        with open(_DICT_PATH, 'r', encoding='utf-8') as _f:
            _PRELOAD_DICT = [line.strip() for line in _f if line.strip()]
        _PRELOAD_DICT.insert(0, 'blank')
        _PRELOAD_DICT.append(' ')
        while len(_PRELOAD_DICT) < 6625:
            _PRELOAD_DICT.append('')
        _PRELOAD_DICT_SIZE = len(_PRELOAD_DICT)
        print(f"[Preload] CTC dict loaded ({_PRELOAD_DICT_SIZE} chars)")
except Exception as e:
    print(f"[WARN] Dict preload skipped: {e}")


# ============================================================
#  ★★★ 配置区（改这里就行）★★★
# ============================================================

# ── 基础路径 ──
# 开发板上模型存放目录，根据实际部署路径修改
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'pp-ocrv4_rapid_onnx')
SAVE_DIR = os.path.join(BASE_DIR, 'photos')

# ── 模型文件名 ──
DET_ONNX = 'ch_PP-OCRv4_det_mobile.onnx'      # 文本检测 ONNX
CLS_ONNX = 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'  # 方向分类 ONNX

# ★ Rec 模型选择 (一行切换 FP32 / INT8) ★
#   FP32: 精度最高, avg_score≈0.95, Rec≈11s
#   INT8: 需要自定义校准集, 官方INT8精度崩坏(见诊断报告)
REC_ONNX = 'ch_PP-OCRv4_rec_mobile.onnx'          # ← FP32 (推荐, 当前最优)
# REC_ONNX = 'ch_PP-OCRv4_rec_mobile_int8.onnx'    # ← INT8 (实验性, 精度崩坏: 31/88框, 0.617score)

# ── 字典文件 ──
DICT_PATH = os.path.join(MODEL_DIR, 'ppocr_keys_v1.txt')

USE_DLC_DET = False    # V3: 固定ORT-CPU
USE_DLC_CLS = False    # V3: 固定ORT-CPU
# ── SNPE/DLC 运行时选择 (★ 真实生效的配置 ★) ──
#   可选: "CPU", "GPU", "DSP"
#   ⚠️ Det 模型实测结论:
#     GPU后端输出噪声(全零), CPU后端固定640x640输出→大图检测质量下降(42框 vs ORT的88框)
#     结论: Det 必须用 ORT-CPU 才能获得与 RapidOCR 一致的检测质量
DET_RUNTIME = "CPU"      # Det: ★ 强制使用 ORT-CPU ★ (DLC固定输出限制检测精度)
CLS_RUNTIME = "GPU"      # Cls: GPU 加速 (轻量模型, GPU可用)
# Rec 固定用 ORT-CPU (MobileOne架构无法转DLC)

# ── ONNX rec 运行时 (保留配置项，当前仅支持 ORT-CPU) ──
REC_ORT_RUNTIME = "CPU"  # Rec 无法转DLC，固定 ONNXRuntime + 多线程

# ── 模式选择（填数字）──
MODE = 1   # 1=图片识别    2=摄像头拍照+识别

# ── 模式1：填图片路径 ──
IMAGE_PATHS = [
    "photos/photo_12.jpg",
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
DET_LIMIT_SIDE_LEN = 736    # 检测输入最大边长 (PC/ONNX模式)
DET_LIMIT_TYPE = 'min'      # 'min'=限制短边  'max'=限制长边
# ★ DLC 模式参数 (DSP输出固定640x640，输入也应匹配)
DLC_DET_LIMIT_SIDE_LEN = 640  # DLC模式用640匹配模型原生分辨率
DLC_DET_LIMIT_TYPE = 'max'    # DLC模式限制长边=640，保持宽高比
DET_THRESH = 0.20           # 二值化阈值
DET_BOX_THRESH = 0.35       # 文本框置信度阈值
DET_UNCLIP_RATIO = 1.6      # 文本框扩展比例
DET_MAX_CANDIDATES = 1000   # 最大候选框数
DET_USE_DILATION = True     # 是否使用膨胀核细化文本区域

# ── Cls 参数 ──
CLS_IMAGE_SHAPE = [3, 48, 192]
CLS_BATCH_NUM = 16        # ★ 增大batch: GPU/DSP下批量推理更快 (原6)
CLS_THRESH = 0.9            # 方向分类置信度阈值（低于此值不旋转）

# ── Rec 参数 ──
REC_IMAGE_SHAPE = [3, 48, 320]
REC_BATCH_NUM = 16          # ★ 增大batch: GPU/DSP下批量推理更快 (原6)

# ── 全局参数 ──
TEXT_SCORE_THRESH = 0.4     # 最终输出最低置信度过滤


# ============================================================
#  SDK 导入（开发板专用）
# ============================================================

def _try_import_sdk():
    """尝试导入开发板 SDK，PC 上运行会优雅降级"""
    try:
        from api_infer import InferenceSession, OnnxContext
        # api_infer 的 runtime 用字符串: "CPU"/"GPU"/"DSP"/"NPU"
        return {
            'InferenceSession': InferenceSession,
            'OnnxContext': OnnxContext,
        }
    except ImportError as e:
        print(f"[WARN] api_infer import failed ({e}) - running in PC simulation mode")
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
    """
    加载 CTC 解码字典
    格式对齐 RapidOCR/PP-OCR:
      index 0 = <blank> (CTC blank token)
      index 1~N = 可见字符
      index N+1 = <space>
    注意: PP-OCRv4 rec 模型输出维度=6625, 原始字典≈6622行
          + blank(1) + space(1) = 6624, 剩余1个为padding类(不应出现)
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f if line.strip()]
    # ★ 对齐 RapidOCR: index 0 插入 blank, 末尾插入 space
    chars.insert(0, 'blank')
    chars.append(' ')
    # ★ 补齐到模型输出维度 (rec model 输出 6625 类)
    #   用空字符串填充(不泄漏可见字符到输出文本)
    while len(chars) < 6625:
        chars.append('')
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

def det_preprocess(img, limit_side_len=736, limit_type='min', pad_to_square=False):
    """
    ★ 文本检测预处理 — 完全对齐 RapidOCR DetPreProcess (ch_ppocr_det/utils.py) ★

    RapidOCR 流程:
      1. 按 limit_type 计算 ratio (min=短边达到 limit_side_len, max=长边限制)
      2. 计算 resize_h, resize_w
      3. ★ round 到 32 的倍数 (不是 padding!)
      4. cv2.resize 到最终尺寸 (无 padding, 无黑边)
      5. 归一化: (img/255 - mean) / std, mean=0.5, std=0.5
      6. HWC → CHW → add batch dim

    pad_to_square (★ DLC 专用 ★):
      DLC Det 输出固定 640×640 prob_map。若输入非方形(如 736×1088)，
      坐标映射比例不一致 → 检测框偏移 → 裁剪错位 → 识别乱码。
      解决: pad_to_square=True 时，将 resize 后图像 pad 到正方形(S×S)，
      DLC 输出 640×640 严格对应 S×S，坐标映射正确。
      返回 resize_info 长度=7: (sq,sq,ratio,orig_h,orig_w,act_h,act_w)
    """
    h, w = img.shape[:2]

    # ★ 对齐 RapidOCR DetPreProcess.resize() 的 ratio 计算逻辑
    if limit_type == 'max':
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0
    else:
        if min(h, w) < limit_side_len:
            if h < w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    # ★ 关键: 先 round 到 32 的倍数, 再 resize (不是 resize 后 padding!)
    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    if resize_h <= 0 or resize_w <= 0:
        return None, None

    # 直接 resize 到目标尺寸 (无 padding!)
    resized = cv2.resize(img, (resize_w, resize_h))

    # ★ DLC 模式: pad 到正方形
    if pad_to_square:
        sq = max(resize_h, resize_w)
        if sq == 0:
            return None, None
        padded = np.zeros((sq, sq, 3), dtype=np.uint8)
        padded[:resize_h, :resize_w] = resized
        work_img = padded
    else:
        work_img = resized

    # 归一化 (PP-OCR 标准归一化: mean=0.5, std=0.5)
    normalized = work_img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (normalized - mean) / std

    # HWC -> CHW -> add batch_dim
    normalized = normalized.transpose((2, 0, 1))
    normalized = np.expand_dims(normalized, axis=0).astype(np.float32)

    if pad_to_square:
        return normalized, (sq, sq, ratio, h, w, resize_h, resize_w)
    else:
        return normalized, (resize_h, resize_w, ratio, h, w)


def det_postprocess(det_result, original_shape, resize_info,
                    thresh=0.30, box_thresh=0.50,
                    unclip_ratio=2.0, max_candidates=1000,
                    use_dilation=False):
    """
    ★ DB 后处理 — 完全对齐 RapidOCR DBPostProcess (ch_ppocr_det/utils.py) ★

    流程 (与 RapidOCR 严格一致):
      1. 二值化 (pred > thresh)
      2. 可选 dilation (2x2 kernel)
      3. findContours → 对每个 contour:
         a. get_mini_boxes → minAreaRect + 4点顺时针排序 + 短边长
         b. [过滤] sside < min_size(3) → skip
         c. box_score_fast → prob_map 内概率均值
         d. [过滤] score < box_thresh → skip
         e. unclip → pyclipper 多边形扩展
         f. get_mini_boxes(again) → 新短边长
         g. [过滤] sside < min_size+2(5) → skip
         h. 坐标映射回原图 (比例缩放 + clip)
      4. filter_det_res → order_points_clockwise + clip + [宽高过滤]
    """
    # 统一提取概率图
    if isinstance(det_result, dict):
        key = list(det_result.keys())[0]
        prob_data = np.array(det_result[key])
    else:
        prob_data = np.array(det_result)

    # 对齐 RapidOCR: pred = pred[:, 0, :, :] 取第0通道
    if prob_data.ndim == 4:
        prob_map = prob_data[0, 0]  # (H, W)
    elif prob_data.ndim == 3:
        prob_map = prob_data[0]     # (H, W)
    elif prob_data.ndim == 2:
        prob_map = prob_data
    else:
        raise ValueError(f"Unexpected det output shape: {prob_data.shape}")

    # ★ DLC pad 模式: resize_info 长度=7，DLC模型输出可能小于padded尺寸(如640 vs 1920)
    #   DLC模型固定输出640x640(DSP编译决定)，需要upscale到padded尺寸再做检测
    #   使用INTER_LINEAR在速度和质量间取平衡
    is_dlc_pad = (len(resize_info) == 7)
    if is_dlc_pad:
        sq, sq2, ratio, orig_h, orig_w, act_h, act_w = resize_info
        out_h, out_w = prob_map.shape[:2]
        if (out_h, out_w) != (sq, sq):
            prob_map = cv2.resize(prob_map, (sq, sq), interpolation=cv2.INTER_LINEAR)
        resize_h, resize_w = sq, sq
        dlc_scale_x = dlc_scale_y = 1.0
    else:
        resize_h, resize_w, ratio, orig_h, orig_w = resize_info
        dlc_scale_x = dlc_scale_y = 1.0

    min_size = 3                       # ★ 对齐 RapidOCR 默认值

    # 二值化 (与 RapidOCR 一致: segmentation = pred > thresh)
    segmentation = prob_map > thresh

    # 可选膨胀（对齐 RapidOCR: 2x2 kernel, use_dilation=True 时执行）
    mask = segmentation.copy()
    if use_dilation:
        kernel = np.array([[1, 1], [1, 1]], np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    # 提取轮廓 (对齐 RapidOCR: RETR_LIST, CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    num_contours = min(len(contours), max_candidates)
    boxes, scores = [], []

    for ci in range(num_contours):
        contour = contours[ci]

        # ★ Step A: get_mini_boxes — 最小外接矩形 + 4点顺时针排列 + 短边长
        points, sside = _get_mini_boxes(contour)

        # ★ Step B: [过滤1] 短边太小
        if sside < min_size:
            continue

        # ★ Step C: box_score_fast — 用 prob_map 计算框内概率均值
        score = _box_score_fast(prob_map, points.reshape(-1, 2))

        # ★ Step D: [过滤2] 分数太低
        if score < box_thresh:
            continue

        # ★ Step E: unclip — pyclipper 多边形精确扩展
        try:
            expanded = _unclip_pyclipper(points, unclip_ratio)
        except Exception:
            continue

        # ★ Step F: unclip 后再做 get_mini_boxes (对齐 RapidOCR!)
        box, sside = _get_mini_boxes(expanded)

        # ★ Step G: [过滤3] unclip 后仍然太小
        if sside < min_size + 2:
            continue

        # ★ Step H: 坐标映射回原图 (对齐 RapidOCR: /resize_* * orig_*)
        if is_dlc_pad:
            # ★ DLC pad模式: padded图像只有[0:act_h, 0:act_w]区域有有效内容
            #   坐标从padded空间→原图需用actual尺寸而非square尺寸
            box[:, 0] = np.clip(np.round(box[:, 0] / act_w * orig_w), 0, orig_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / act_h * orig_h), 0, orig_h)
        else:
            box[:, 0] = np.clip(np.round(box[:, 0] / resize_w * orig_w), 0, orig_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / resize_h * orig_h), 0, orig_h)

        boxes.append(box)
        scores.append(score)

    # ★ Step I: filter_det_res — 最终过滤 (对齐 RapidOCR)
    boxes, scores = _filter_det_res(boxes, scores, orig_h, orig_w)

    return boxes, scores


def _get_mini_boxes(contour):
    """
    ★ 最小外接矩形 + 4点顺时针排列 — 对齐 RapidOCR DBPostProcess.get_mini_boxes ★

    返回: (box, short_side_length)
      box: 4个点, 顺时针 [tl, tr, br, bl], dtype=float32
      short_side: min(width, height) of the minAreaRect
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    # 左边两个点: 按 y 排序 → index_1=上(tl), index_4=下(bl)
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0

    # 右边两个点: 按 y 排序 → index_2=上(tr), index_3=下(br)
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    box = np.array([
        points[index_1],
        points[index_2],
        points[index_3],
        points[index_4],
    ], dtype=np.float32)

    return box, min(bounding_box[1])


def _box_score_fast(bitmap, _box):
    """
    ★ 快速计算框内概率均值 — 对齐 RapidOCR DBPostProcess.box_score_fast ★

    用最小外接矩形的4个顶点多边形在 bitmap 上做 mask mean
    """
    h, w = bitmap.shape[:2]
    box = _box.copy()

    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box_local = box.copy()
    box_local[:, 0] -= xmin
    box_local[:, 1] -= ymin
    cv2.fillPoly(mask, box_local.reshape(1, -1, 2).astype(np.int32), 1)

    return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask=mask)[0])


def _unclip_pyclipper(box, unclip_ratio):
    """
    ★ pyclipper 多边形扩展 — 对齐 RapidOCR DBPostProcess.unclip ★
    """
    if not _has_pyclipper:
        raise ImportError("pyclipper/shapely not available")

    poly = ShapelyPolygon(box)
    distance = poly.area * unclip_ratio / poly.length

    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)

    return np.array(expanded[0], dtype=np.float32).reshape((-1, 1, 2))


def _filter_det_res(dt_boxes, scores, img_height, img_width):
    """
    ★ 最终候选框过滤 — 对齐 RapidOCR DBPostProcess.filter_det_res ★

    对每个 box:
      1. order_points_clockwise (确保顺时针 tl,tr,br,bl)
      2. clip 到图像范围内 [0, img_width-1] x [0, img_height-1]
      3. [过滤] 宽<=3 或 高<=3 的框丢弃
    """
    dt_boxes_new = []
    new_scores = []
    for box, score in zip(dt_boxes, scores):
        box = _order_points_clockwise(box.astype(np.float32))

        # ★ 对齐 RapidOCR clip_det_res: 边界是 img_width-1 / img_height-1
        for pno in range(4):
            box[pno, 0] = float(min(max(int(round(box[pno, 0])), 0), img_width - 1))
            box[pno, 1] = float(min(max(int(round(box[pno, 1])), 0), img_height - 1))

        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))

        # ★ [过滤4] 宽或高 <= 3 丢弃
        if rect_width <= 3 or rect_height <= 3:
            continue

        dt_boxes_new.append(box)
        new_scores.append(score)

    return dt_boxes_new, new_scores


def unclip_box(box, unclip_ratio=2.0):
    """
    ★ 多边形精确扩展 — 对齐 RapidOCR (shapely + pyclipper) ★
    
    使用 shapely 计算面积/周长，pyclipper 做多边形偏移
    """
    if unclip_ratio <= 1.0:
        return box

    try:
        if not _has_pyclipper:
            raise ImportError("pyclipper/shapely not available")

        poly = ShapelyPolygon(box)
        # shapely: area + length (周长)
        distance = poly.area * unclip_ratio / poly.length
        
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.astype(np.float64).tolist(), 
                       pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        # 取第一个（最大的）多边形结果
        return np.array(expanded[0], dtype=np.float32).reshape(-1, 2)
        
    except ImportError:
        pass  # 降级到中心扩展
    except Exception as e:
        print(f"[WARN] pyclipper unclip failed: {e}, fallback to center expand")
        pass

    # 降级方案: 从中心点扩展
    center_x = box[:, 0].mean()
    center_y = box[:, 1].mean()
    expanded = box.copy()
    expanded[:, 0] = center_x + (box[:, 0] - center_x) * unclip_ratio
    expanded[:, 1] = center_y + (box[:, 1] - center_y) * unclip_ratio
    return expanded


def _order_points_clockwise(pts):
    """
    ★ 将 4 个点按顺时针排列 (tl, tr, br, bl) — 对齐 RapidOCR ★
    
    tl=左上, tr=右上, br=右下, bl=左下
    
    这对 get_rotate_crop_image 的透视变换裁剪至关重要：
      顺序错误会导致文本区域被旋转/镜像
    """
    pts = np.array(pts, dtype=np.float32)
    # 按 x 坐标排序 → 左边2个点 + 右边2个点
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    # 左半部分按 y 排序: 上→下 = tl, bl
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    # 右半部分按 y 排序: 上→下 = tr, br
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tl, bl) = left_most
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_rotate_crop_image(img, box):
    """★ 裁剪并矫正旋转文本区域 — 完全对齐 RapidOCR process_img.py ★

    关键差异（已对齐）:
      1. 使用 INTER_CUBIC 插值（非默认的 LINEAR）→ 更清晰的裁剪图像
      2. 高度/宽度 >= 1.5 时自动 rot90 修正 → 竖向文字正确识别
      3. borderMode=BORDER_REPLICATE（与 RapidOCR 一致）
    """
    points = box.astype(np.float32)
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]))
    )
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]))
    )

    pts_std = np.array([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,          # ★ 对齐 RapidOCR: CUBIC 插值
    )
    # ★ 对齐 RapidOCR: 竖向文字自动旋转修正 (高度>>宽度时说明检测框是竖着的)
    dst_img_h, dst_img_w = dst_img.shape[:2]
    if dst_img_h * 1.0 / dst_img_w >= 1.5:
        dst_img = np.rot90(dst_img)

    return dst_img


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
    """
    ★ 文字识别预处理 — 对齐 RapidOCR resize_norm_img ★
    
    关键修复（原版本有3个致命bug）:
      1. padding 值: 128(灰) → 0(黑)  np.full→np.zeros
      2. 策略: min(h,w)双限制 → 固定高度=48，宽度按比例
      3. 宽度: 固定320 → 动态宽度(按batch内最大wh_ratio)
    """
    imgC, imgH, imgW = 3, target_shape[0], target_shape[1]
    max_wh_ratio = imgW / imgH

    # 第一遍：计算 batch 中最大宽高比
    for img in images:
        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)

    actual_imgW = int(imgH * max_wh_ratio)

    batch = []
    for img in images:
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > actual_imgW:
            resized_w = actual_imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        # ★ 固定高度=imgH(48)，宽度按比例
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype(np.float32)
        # 先转 CHW 再归一化 (与 RapidOCR 一致)
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        # ★ pad=0 (黑色背景)
        padding_im = np.zeros((imgC, imgH, actual_imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        batch.append(padding_im)

    return np.stack(batch, axis=0)


def ctc_decode_greedy(probs, dict_chars):
    """
    ★ CTC Greedy 解码 — 对齐 RapidOCR CTCLabelDecode ★
    
    修复:
      - 字典 index 0 = blank (已在 load_dict 中插入)
      - score 使用 preds.max(axis=2) 即每个时间步的最大概率
      - 去重: 连续相同字符只保留一个
      - 忽略 blank token (index=0)
    """
    texts = []
    text_scores = []

    # probs shape: (batch, seq_len, num_classes)
    if probs.ndim == 3:
        batch_size = probs.shape[0]
    else:
        probs = probs[np.newaxis]
        batch_size = 1

    for b in range(batch_size):
        pred_idx = np.argmax(probs[b], axis=1)
        pred_prob = np.max(probs[b], axis=1)

        # 去重：连续相同字符只保留第一个
        selection = np.ones(len(pred_idx), dtype=bool)
        selection[1:] = pred_idx[1:] != pred_idx[:-1]

        # 忽略 blank token (index=0)
        selection &= (pred_idx != 0)

        if selection.any():
            conf_list = pred_prob[selection].tolist()
            conf_list = [round(c, 5) for c in conf_list]
            selected_idx = pred_idx[selection]
            # 安全边界检查 + 过滤空字符串(字典padding占位)
            valid_mask = selected_idx < len(dict_chars)
            if not valid_mask.all():
                selected_idx = selected_idx[valid_mask]
                conf_list = [c for c, v in zip(conf_list, valid_mask) if v]
            char_list = [dict_chars[idx] for idx in selected_idx if dict_chars[idx]]  # ★ 过滤空字符串
            # 同步更新 conf_list
            if len(char_list) != len([dict_chars[idx] for idx in selected_idx]):
                # 重新计算 conf: 只保留非空字符的置信度
                _valid_chars = [(dict_chars[idx], c) for idx, c in zip(selected_idx, conf_list) if idx < len(dict_chars) and dict_chars[idx]]
                if _valid_chars:
                    char_list = [c[0] for c in _valid_chars]
                    conf_list = [c[1] for c in _valid_chars]
            text = ''.join(char_list)
            avg_score = float(np.mean(conf_list))
        else:
            text = ''
            avg_score = 0.0

        texts.append(text)
        text_scores.append(avg_score)

    return texts, text_scores


def _ctc_single(prob_seq, dict_chars):
    """单条 CTC 序列解码（保留兼容旧接口）"""
    pred_idx = np.argmax(prob_seq, axis=1)
    prev_idx = -1
    chars = []
    scores = []
    for t, idx in enumerate(pred_idx):
        if idx != prev_idx:
            if idx > 0 and idx < len(dict_chars):
                chars.append(dict_chars[idx])
                scores.append(float(prob_seq[t, idx]))
            prev_idx = idx
    text = ''.join(chars)
    avg_score = float(np.mean(scores)) if scores else 0.0
    return text, avg_score


# ============================================================
#  ★ 核心: OCR 引擎 (DLC + ONNX 混合推理版) ★
# ============================================================

class OCRBoardEngine:
    """
    开发板 OCR 引擎

    推理后端 (配置区 DET_RUNTIME / CLS_RUNTIME 控制):
      - Det: DLC (SNPE) → GPU/DSP/CPU 硬件加速  ★ GPU 已验证可用
      - Cls: DLC (SNPE) → GPU/DSP/CPU 硬件加速  ★ GPU 已验证可用
      - Rec: ONNX (ORT) → CPU 多线程           (MobileOne 无法转 DLC)

    参考实现: yolo人体检测/工程源码/main.py (同 SDK, runtime="GPU" 已跑通)
    """

    def __init__(self):
        self.det_session = None
        self.cls_session = None
        self.rec_session = None
        self.dict_chars = None
        self.dict_size = 0
        self._load_time = 0
        self._is_pc_mode = (_SDK is None)
        # ★ 后端类型标志: True=SDK(InferenceSession), False=ORT(InferenceSession) ★
        self._det_is_sdk = False
        self._cls_is_sdk = False
        self._rec_is_sdk = False

    def init_model(self):
        """加载所有模型"""
        t0 = time.perf_counter()

        print("=" * 55)
        print("  Board OCR Engine - Loading Models")
        print("=" * 55)
        # V3: 全部固定 ORT-CPU (已验证最优, 见诊断报告)
        det_backend = "ORT-CPU (full-resolution)"
        cls_backend = "ORT-CPU"
        rec_backend = "ORT-CPU"
        print(f"  Det: {DET_ONNX} ({det_backend})")
        print(f"  Cls: {CLS_ONNX} ({cls_backend})")
        print(f"  Rec: {REC_ONNX} ({rec_backend})")

        # ★ 使用预加载的 CTC 字典 (模块导入时已读取) ★
        if _PRELOAD_DICT is not None:
            self.dict_chars = _PRELOAD_DICT[:]
            self.dict_size = _PRELOAD_DICT_SIZE
            print(f"  Dict: preloaded ({self.dict_size} chars)")
        else:
            # fallback: 字典未预加载时从文件读取
            self.dict_chars, self.dict_size = load_dict(DICT_PATH)
            print(f"  Dict: loaded from file ({self.dict_size} chars)")

        if self._is_pc_mode:
            self._init_pc_fallback()
        else:
            self._init_board_models()

        self._load_time = time.perf_counter() - t0
        print(f"\n  All models loaded! ({self._load_time:.3f}s)\n")

    def _init_board_models(self):
        """开发板模式: Det/Cls → DLC+SNPE+GPU硬件加速, Rec → ORT-CPU多线程

        ★ 参考实现: yolo人体检测/工程源码/main.py (同SDK, GPU已验证)
        ★ 调用链: api_infer.InferenceSession → Initialize() → Execute()
        """

        det_onnx_path = os.path.join(MODEL_DIR, DET_ONNX)
        cls_onnx_path = os.path.join(MODEL_DIR, CLS_ONNX)
        rec_onnx_path = os.path.join(MODEL_DIR, REC_ONNX)

        # ════════════════════════════════════════════
        #  Det: ★ ORT-CPU (全分辨率输出, 匹配 RapidOCR 精度) ★
        #   原因: DLC 固定 640×640 输出 → 大图检测仅 42 框(ORT 可达 88 框)
        #         GPU 后端输出噪声, CPU 后端分辨率不足 → 统一使用 ORT-CPU
        # ════════════════════════════════════════════
        if ort is not None:
            print("  Loading Det (ONNX / ORT-CPU full-resolution)...")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 2
            self.det_session = ort.InferenceSession(det_onnx_path, so)
            self._det_is_sdk = False
            print(f"  [OK] Det loaded (ORT-CPU, full-res output)")
        else:
            raise ImportError("Det requires onnxruntime (not available)")

        # ════════════════════════════════════════════
        #  Cls: ★ ORT-CPU (DLC-GPU误旋转导致19%文本丢失, 已禁用) ★
        # ════════════════════════════════════════════
        if ort is not None:
            print("  Loading Cls (ONNX / ORT-CPU)...")
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            self.cls_session = ort.InferenceSession(cls_onnx_path, so)
            self._cls_is_sdk = False
            print("  [OK] Cls loaded (ORT-CPU)")
        else:
            raise ImportError("Cls requires onnxruntime (not available)")

        # ════════════════════════════════════════════
        #  Rec: ONNX → ORT → CPU 多线程 (MobileOne无法转DLC)
        # ════════════════════════════════════════════
        if ort is None:
            raise ImportError("Rec requires onnxruntime (not available)")
        print(f"  Loading Rec (ONNX / ORT-CPU multi-thread)...")
        so_rec = ort.SessionOptions()
        so_rec.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so_rec.intra_op_num_threads = 4
        self.rec_session = ort.InferenceSession(rec_onnx_path, so_rec)
        self._rec_is_sdk = False   # Rec 固定用 ORT
        print("  [OK] Rec loaded (ORT-CPU)")

    def _init_pc_fallback(self):
        """PC 回退模式: 全部用 ONNXRuntime (方便本地测试)"""
        if ort is None:
            raise ImportError("onnxruntime not available")
        print("\n  [PC Mode] Using ONNXRuntime for all models...")

        det_path = os.path.join(MODEL_DIR, DET_ONNX)
        cls_path = os.path.join(MODEL_DIR, CLS_ONNX)
        rec_path = os.path.join(MODEL_DIR, REC_ONNX)

        if not os.path.exists(det_path):
            det_path = os.path.join(MODEL_DIR,
                        [f for f in os.listdir(MODEL_DIR) if 'det' in f.lower() and f.endswith('.onnx')][0]
                        if any('det' in f.lower() for f in os.listdir(MODEL_DIR)) else '')
        if not os.path.exists(cls_path):
            cls_path = os.path.join(MODEL_DIR,
                        [f for f in os.listdir(MODEL_DIR) if 'cls' in f.lower() and f.endswith('.onnx')][0]
                        if any('cls' in f.lower() for f in os.listdir(MODEL_DIR)) else '')

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 2
        self.det_session = ort.InferenceSession(det_path, so)
        self.cls_session = ort.InferenceSession(cls_path, so)
        self.rec_session = ort.InferenceSession(rec_path, so)
        # PC fallback: 全部 ORT-CPU
        self._det_is_sdk = False
        self._cls_is_sdk = False
        self._rec_is_sdk = False
        print(f"  [OK] PC Fallback: det={os.path.basename(det_path)} "
              f"cls={os.path.basename(cls_path)} rec={os.path.basename(rec_path)} (4 threads)")

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
        # ===== Stage 1: Text Detection (ORT-CPU, 全分辨率输出) =====
        t0 = time.perf_counter()
        det_input, resize_info = det_preprocess(
            img,
            limit_side_len=DET_LIMIT_SIDE_LEN,
            limit_type=DET_LIMIT_TYPE,
            pad_to_square=False,
        )
        if det_input is None:
            return {'texts': [], 'scores': [], 'boxes': [],
                    'count': 0, 'avg_score': 0, 'elapsed': 0,
                    'elapsed_det': 0, 'elapsed_cls': 0, 'elapsed_rec': 0}
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
        # ★ 对齐 RapidOCR: 只有 cls_score >= CLS_THRESH 时才信任方向分类结果
        #   低置信度的方向判断会被忽略，保持原图不旋转（避免误翻转导致乱码）
        rotated_crops = []
        for crop, angle, cls_sc in zip(crops, all_angles, all_cls_scores):
            if angle == 1 and cls_sc >= CLS_THRESH:  # 高置信度才执行180度旋转
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

        # 组装最终结果
        # ★ 对齐 RapidOCR filter_by_text_score: 直接用 rec_score 过滤 (不做 text*box 乘积)
        valid_texts, valid_scores, valid_boxes = [], [], []
        for txt, sc, bs, box in zip(all_texts, all_text_scores, box_scores, boxes):
            # ★ 对齐 RapidOCR: 仅用 rec_score 与 text_score_thresh 比较
            if sc >= TEXT_SCORE_THRESH:
                valid_texts.append(txt)
                valid_scores.append(sc)   # ★ 直接用 rec CTC 解码的置信度
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
        """执行 Det 推理 — 自动适配 SDK(GPU/DSP) 或 ORT(CPU) 后端"""
        input_feed = {'x': input_data.astype(np.float32)}
        output_names = ['sigmoid_0.tmp_0']

        if self._det_is_sdk:
            # ★ SDK InferenceSession: DLC → SNPE → GPU/DSP/CPU 硬件加速
            result = self.det_session.Execute(output_names, input_feed)
            val = np.array(result.get('sigmoid_0.tmp_0', []))
            # 处理 SDK 返回的扁平化输出
            if val.ndim == 1:
                total = val.shape[0]
                side = int(round(total ** 0.5))
                if side * side == total:
                    val = val.reshape(1, 1, side, side)
            return {output_names[0]: val}
        else:
            # ★ ONNXRuntime: CPU 多线程
            out = self.det_session.run(output_names, input_feed)
            return {output_names[0]: out[0]}

    def _run_cls(self, input_data):
        """执行 Cls 推理 — SDK模式逐样本(不支持batch), ORT模式批量"""
        output_names = ['save_infer_model/scale_0.tmp_1']

        if self._cls_is_sdk:
            # ★ SDK InferenceSession: 不支持 batch 推理，逐样本循环 ★
            #   (SNPE Execute 的输出 buffer 按单样本 shape 分配)
            batch = input_data.shape[0]
            all_preds = []
            for i in range(batch):
                single_input = input_data[i:i+1].astype(np.float32)
                input_feed = {'x': single_input}
                result = self.cls_session.Execute(output_names, input_feed)
                val = np.array(result.get(output_names[0], []))
                all_preds.append(val.flatten()[:2])  # 2分类概率 [正向, 180°翻转]
            return {output_names[0]: np.array(all_preds)}  # (B, 2)
        else:
            # ★ ONNXRuntime: 原生支持批量
            input_feed = {'x': input_data.astype(np.float32)}
            out = self.cls_session.run(output_names, input_feed)
            return {output_names[0]: out[0]}

    def _run_rec(self, input_data):
        """执行 Rec 推理 — 固定使用 ORT-CPU (MobileOne无法转DLC)
        
        ★ V3修复: SDK路径的dict_size已修正为模型实际输出维度6625
           旧bug: self.dict_size=6622 ≠ 模型输出=6625 → reshape错位 → CTC乱码
        """
        input_feed = {'x': input_data.astype(np.float32)}
        output_names = ['softmax_11.tmp_0']

        if self._rec_is_sdk:
            # SDK OnnxContext 路径
            result = self.rec_session.Execute(output_names, input_feed)
            val = result.get('softmax_11.tmp_0', [])
            batch = input_data.shape[0]
            # ★ V3修复: 用模型实际输出维度(6625)而非字典大小(6622)
            REC_OUTPUT_DIM = 6625  # PP-OCRv4 rec_mobile 固定输出维度
            seq_len = len(val) // batch // REC_OUTPUT_DIM
            raw = np.array(val).reshape(batch, seq_len, REC_OUTPUT_DIM)
            return raw[:, :, :self.dict_size]  # 裁剪到字典大小
        else:
            # ONNXRuntime CPU 多线程 (默认路径)
            out = self.rec_session.run(output_names, input_feed)
            return out[0]  # (B, T, C)

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
