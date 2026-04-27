"""
诊断: 为什么同样参数, _debug_det_params.py 得 88框, ocr_board.py 得 71框?

逐步比对:
  1. det_preprocess 输出是否一致
  2. 模型推理输出 shape 是否一致
  3. det_postprocess 输出是否一致
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_board import (
    load_image, det_preprocess, det_postprocess,
    MODEL_DIR, DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE,
    DET_THRESH, DET_BOX_THRESH, DET_UNCLIP_RATIO,
    DET_MAX_CANDIDATES, DET_USE_DILATION,
    OCRBoardEngine,
)
import onnxruntime as ort

# ── 加载图片 ──
img = load_image("photos/photo_1.jpg")
print(f"Image: {img.shape}")

# ── 方法A: 直接调用函数 (跟 _debug_det_params.py 完全一样) ──
print("\n" + "=" * 60)
print("方法A: 直接调用 det_preprocess + det_postprocess")
print("=" * 60)

det_path = os.path.join(MODEL_DIR,
    [f for f in os.listdir(MODEL_DIR) if 'det' in f.lower() and f.endswith('.onnx')][0])
so = ort.SessionOptions()
so.intra_op_num_threads = 4
session_a = ort.InferenceSession(det_path, so)

input_a, resize_info_a = det_preprocess(
    img, limit_side_len=DET_LIMIT_SIDE_LEN,
    limit_type=DET_LIMIT_TYPE, pad_to_square=False)
print(f"A preprocess: input shape={input_a.shape}, resize_info={resize_info_a}")

in_name = session_a.get_inputs()[0].name
out_name = session_a.get_outputs()[0].name
output_a = session_a.run([out_name], {in_name: input_a})[0]
print(f"A model out:  shape={output_a.shape}, min={output_a.min():.4f}, max={output_a.max():.4f}")

boxes_a, scores_a = det_postprocess(
    {out_name: output_a}, img.shape, resize_info_a,
    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
    use_dilation=DET_USE_DILATION,
)
print(f"A postprocess: {len(boxes_a)} boxes")

# ── 方法B: 通过 OCRBoardEngine.recognize() ──
print("\n" + "=" * 60)
print("方法B: 通过 OCRBoardEngine (完整 pipeline)")
print("=" * 60)

engine = OCRBoardEngine()
engine.init_model()
print(f"B _det_is_sdk={engine._det_is_sdk}")

# 复用 engine 的预处理/推理/后处理, 但单独调用
input_b, resize_info_b = det_preprocess(
    img, limit_side_len=DET_LIMIT_SIDE_LEN,
    limit_type=DET_LIMIT_TYPE, pad_to_square=False)
print(f"B preprocess: input shape={input_b.shape}, resize_info={resize_info_b}")

# 检查预处理是否完全一致
preprocess_match = (
    np.array_equal(input_a, input_b) and
    resize_info_a == resize_info_b
)
print(f"  Preprocess 一致? {preprocess_match}")
if not preprocess_match:
    diff = np.abs(input_a.astype(float) - input_b.astype(float))
    print(f"  Input 差异: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

output_b_raw = engine._run_det(input_b)
out_key = list(output_b_raw.keys())[0]
output_b = output_b_raw[out_key]
print(f"B model out:  shape={output_b.shape}, min={output_b.min():.4f}, max={output_b.max():.4f}")

# 检查模型输出是否一致
model_match = np.array_equal(output_a, output_b)
print(f"  Model output 一致? {model_match}")
if not model_match:
    print(f"  A shape={output_a.shape} vs B shape={output_b.shape}")
    if output_a.shape == output_b.shape:
        diff = np.abs(output_a.astype(float) - output_b.astype(float))
        print(f"  Output 差异: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

boxes_b, scores_b = det_postprocess(
    output_b_raw, img.shape, resize_info_b,
    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
    use_dilation=DET_USE_DILATION,
)
print(f"B postprocess: {len(boxes_b)} boxes")

# ── 对比 ──
print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)
print(f"  方法A (直接函数): {len(boxes_a)} boxes")
print(f"  方法B (OCRBoard): {len(boxes_b)} boxes")
print(f"  差异: {len(boxes_a) - len(boxes_b)} boxes")

if len(boxes_a) != len(boxes_b):
    # 尝试不同参数看看是不是参数问题
    print("\n  --- 尝试其他参数组合 ---")
    for sl in [640, 736, 960, 1088]:
        for lt in ['min', 'max']:
            inp, ri = det_preprocess(img, limit_side_len=sl, limit_type=lt)
            out = engine._run_det(inp)
            bx, sc = det_postprocess(out, img.shape, ri,
                                       thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
                                       unclip_ratio=DET_UNCLIP_RATIO,
                                       max_candidates=DET_MAX_CANDIDATES,
                                       use_dilation=DET_USE_DILATION)
            tag = " ← 当前配置" if (sl == DET_LIMIT_SIDE_LEN and lt == DET_LIMIT_TYPE) else ""
            print(f"    side={sl:>4} type={lt:>3} → {len(bx)} boxes{tag}")
