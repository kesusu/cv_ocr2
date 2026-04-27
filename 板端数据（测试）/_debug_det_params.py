"""
Priority 1: ORT-Det 参数调优 — 提升检测数量 71→88

目标: 找到与 RapidOCR(88框)一致的 Det 参数配置
基准: ocr_workflow_onnx.py (RapidOCR) -> 88 boxes
当前: ocr_board.py (ORT-CPU)     -> 71 boxes

扫描维度:
  1. limit_side_len: [736, 960, 1088, 1280, 1480]  — 输入分辨率
  2. limit_type:     ['min', 'max']                   — 限制短边/长边
  3. thresh:         [0.15, 0.18, 0.20, 0.25, 0.30]  — 二值化阈值
  4. box_thresh:     [0.25, 0.30, 0.35, 0.40]          — 框置信度阈值
"""

import os
import sys
import time
import cv2
import numpy as np

# ── 导入 ocr_board 的预处理/后处理函数 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_board import (
    load_image, det_preprocess, det_postprocess,
    MODEL_DIR, DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION,
)

import onnxruntime as ort


def load_det_model():
    """加载 Det ONNX 模型"""
    det_path = os.path.join(MODEL_DIR,
        [f for f in os.listdir(MODEL_DIR) if 'det' in f.lower() and f.endswith('.onnx')][0])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    session = ort.InferenceSession(det_path, so)
    print(f"[OK] Det model: {os.path.basename(det_path)}")
    return session


def run_det_once(session, img, limit_side_len, limit_type, thresh, box_thresh):
    """单次 Det 推理 + 后处理, 返回框数"""
    # 预处理
    det_input, resize_info = det_preprocess(
        img, limit_side_len=limit_side_len,
        limit_type=limit_type, pad_to_square=False)

    # 推理
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    det_output = session.run([output_name], {input_name: det_input})[0]

    # 后处理
    boxes, scores = det_postprocess(
        det_output, img.shape, resize_info,
        thresh=thresh, box_thresh=box_thresh,
        unclip_ratio=DET_UNCLIP_RATIO,
        max_candidates=DET_MAX_CANDIDATES,
        use_dilation=DET_USE_DILATION,
    )
    return len(boxes), boxes, scores


def main():
    print("=" * 70)
    print("  Priority 1: ORT-Det 参数扫描 — 目标 88 框")
    print("=" * 70)

    # 加载模型
    session = load_det_model()

    # 加载测试图片
    img_path = "photos/photo_1.jpg"
    if not os.path.exists(img_path):
        print(f"[ERROR] 图片不存在: {img_path}")
        return
    img = load_image(img_path)
    h, w = img.shape[:2]
    print(f"  图片: {img_path} ({w}x{h})")
    print()

    # ── Phase A: 粗扫 — 固定 thresh=0.20, box_thresh=0.35, 扫 side_len + type ──
    print("=" * 70)
    print("  Phase A: 扫描 limit_side_len + limit_type")
    print("           (固定 thresh=0.20, box_thresh=0.35)")
    print("=" * 70)

    side_lens = [640, 736, 800, 960, 1088, 1280, 1480, 1920]
    limit_types = ['min', 'max']

    results_a = []
    for lt in limit_types:
        for sl in side_lens:
            try:
                n, _, _ = run_det_once(session, img, sl, lt, 0.20, 0.35)
                results_a.append((sl, lt, n))
                tag = "★ TARGET!" if n == 88 else ("close" if 80 <= n <= 95 else "")
                print(f"  side={sl:>4}  type={lt:>3}  → {n:>3} boxes  {tag}")
            except Exception as e:
                results_a.append((sl, lt, -1))
                print(f"  side={sl:>4}  type={lt:>3}  → ERROR: {e}")

    print()

    # ── Phase B: 在最优 side_len 附近, 扫 thresh + box_thresh ──
    print("=" * 70)
    print("  Phase B: 在最优 side_len 附近扫描 thresh × box_thresh")
    print("=" * 70)

    # 选 Phase A 中最接近 88 的 top-3 配置
    results_a.sort(key=lambda x: abs(x[2] - 88))
    top3 = results_a[:3]
    print(f"  Phase A Top-3 (closest to 88): {top3}")
    print()

    threshes = [0.15, 0.18, 0.20, 0.25, 0.30]
    box_threshes = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    best_config = None
    best_diff = 999

    for base_sl, base_lt, _ in top3:
        print(f"  --- base: side_len={base_sl}, limit_type={base_lt} ---")
        for th in threshes:
            row_vals = []
            for bt in box_threshes:
                try:
                    n, _, _ = run_det_once(session, img, base_sl, base_lt, th, bt)
                    row_vals.append(n)
                    diff = abs(n - 88)
                    if diff < best_diff:
                        best_diff = diff
                        best_config = (base_sl, base_lt, th, bt, n)
                except Exception as e:
                    row_vals.append(-1)

            # 打印行
            line = f"    th={th:.2f}|"
            for v in row_vals:
                if v == 88:
                    line += f" ★{v:>3}★|"
                elif 80 <= v <= 95:
                    line += f" [{v:>3}]|"
                elif v >= 0:
                    line += f"  {v:>3} |"
                else:
                    line += "  ERR |"
            print(line)
        print()

    # ── 结果汇总 ──
    print("=" * 70)
    print("  ★ 最优配置 ★")
    print("=" * 70)
    if best_config:
        sl, lt, th, bt, n = best_config
        print(f"  limit_side_len = {sl}")
        print(f"  limit_type     = '{lt}'")
        print(f"  thresh         = {th}")
        print(f"  box_thresh     = {bt}")
        print(f"  => {n} boxes (目标88, 差{abs(n-88)})")

        if n == 88:
            print("\n  ★★★ 恰好达到 88 框! 可直接更新 ocr_board.py 配置 ★★★")
        elif n > 85:
            print(f"\n  ★ 接近目标! 可微调 thresh/box_thresh 精确匹配 ★")
    else:
        print("  未找到有效配置!")

    print()


if __name__ == '__main__':
    main()
