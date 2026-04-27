#!/usr/bin/env python3
"""
Rec 线程数优化: 扫描 intra_op_num_threads 对速度的影响
目标: 在 8 核 CPU 上找到最优线程配置
"""
import sys, os, time, cv2, numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    OCRBoardEngine, load_image,
    det_preprocess, det_postprocess,
    get_rotate_crop_image, cls_preprocess, cls_postprocess,
    rec_preprocess, ctc_decode_greedy,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_THRESH, DET_BOX_THRESH,
    DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION,
    CLS_IMAGE_SHAPE, CLS_BATCH_NUM,
    REC_IMAGE_SHAPE, MODEL_DIR, CLS_THRESH, TEXT_SCORE_THRESH,
)

REC_ONNX = 'ch_PP-OCRv4_rec_mobile.onnx'

def main():
    print("=" * 70)
    print("  Rec 线程数优化: intra_op_num_threads 扫描")
    print("=" * 70)

    # ── 准备数据 (Det+Cls 只跑一次) ──
    engine = OCRBoardEngine()
    engine.init_model()
    img = load_image('photos/photo_1.jpg')

    # Det
    t0 = time.perf_counter()
    det_input, resize_info = det_preprocess(
        img, DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, pad_to_square=False)
    det_output = engine._run_det(det_input)
    det_t = time.perf_counter() - t0
    boxes, box_scores = det_postprocess(
        det_output, img.shape, resize_info,
        thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
        unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
        use_dilation=DET_USE_DILATION,
    )
    crops = [get_rotate_crop_image(img, box) for box in boxes]

    # Cls
    ci_all = cls_preprocess(crops, tuple(CLS_IMAGE_SHAPE[1:]))
    co_all = engine._run_cls(ci_all)
    angles_all, cls_scores_all = cls_postprocess(co_all, CLS_THRESH)

    rotated_crops = []
    for crop, angle, cls_sc in zip(crops, angles_all, cls_scores_all):
        if angle == 1 and cls_sc >= CLS_THRESH:
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        rotated_crops.append(crop)

    total_boxes = len(rotated_crops)
    rec_input_batch = rec_preprocess(rotated_crops, tuple(REC_IMAGE_SHAPE[1:]))
    rec_onnx_path = os.path.join(MODEL_DIR, REC_ONNX)
    ort_in_name = None  # 稍后获取
    ort_out_name = None

    print(f"\n  Det: {det_t:.3f}s, Boxes: {total_boxes}")
    print(f"  Rec input shape: {rec_input_batch.shape}")

    # ── 测试不同线程数 ──
    thread_tests = [1, 2, 4, 6, 8]

    print(f"\n{'='*75}")
    print(f"  {'Threads':>7} | {'Batch':>5} | {'Calls':>5} | {'RecTime':>8} | "
          f"{'PerCall':>8} | {'AvgScore':>9} | {'Texts':>6}")
    print("-" * 75)

    best_time = 999
    best_cfg = None

    for n_threads in thread_tests:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = n_threads
        session = ort.InferenceSession(rec_onnx_path, so)

        if ort_in_name is None:
            ort_in_name = session.get_inputs()[0].name
            ort_out_name = session.get_outputs()[0].name

        # Warmup
        _ = session.run([ort_out_name], {ort_in_name: rec_input_batch[:1]})

        # 实测: 不同 batch size
        batch_sizes_to_test = [1, 4, 8, 16, 32] if n_threads == thread_tests[-1] else [16]
        
        for bs in batch_sizes_to_test:
            t0 = time.perf_counter()
            all_texts, all_text_scores = [], []
            n_calls = 0
            for start in range(0, total_boxes, bs):
                batch_data = rec_input_batch[start:start + bs]
                out = session.run([ort_out_name], {ort_in_name: batch_data})
                texts, scores = ctc_decode_greedy(out[0], engine.dict_chars)
                all_texts.extend(texts)
                all_text_scores.extend(scores)
                n_calls += 1
            rec_t = time.perf_counter() - t0

            per_call = rec_t / max(n_calls, 1)
            valid = [(t, s) for t, s in zip(all_texts, all_text_scores) if s >= TEXT_SCORE_THRESH and t.strip()]
            n_valid = len(valid)
            avg_sc = float(np.mean([s for _, s in valid])) if valid else 0

            tag = ""
            is_best_for_thread = False
            if rec_t < best_time:
                best_time = rec_t
                best_cfg = (n_threads, bs, rec_t, avg_sc, n_valid, per_call, n_calls)
                is_best_for_thread = True
                tag = " ★★ BEST"

            # 只有最佳batch或默认batch才打印详细行
            if bs == 16 or is_best_for_thread:
                bs_tag = f"(bs={bs})" if len(batch_sizes_to_test) > 1 else ""
                print(f"  {n_threads:>7} | {bs:>5d} | {n_calls:>5d} | {rec_t:>7.3f}s | "
                      f"{per_call:>7.3f}s | {avg_sc:>9.4f} | {n_valid:>6d}{tag}{bs_tag}")

        del session

    # ── 最终结论 ──
    nt, bs, rt, sc, nv, pc, nc = best_cfg
    total_est = det_t + 0.35 + rt
    print(f"\n{'='*75}")
    print(f"  ★ 最优配置:")
    print(f"     intra_op_num_threads = {nt}")
    print(f"     REC_BATCH_NUM         = {bs}")
    print(f"     Rec耗时               = {rt:.3f}s ({nc}次调用, 均{pc:.3f}s/次)")
    print(f"     Texts                 = {nv}, AvgScore = {sc:.4f}")
    print(f"     预估Total             ≈ {total_est:.2f}s (Det={det_t:.2f} + Cls≈0.35 + Rec={rt:.2f})")
    print(f"     比当前(threads=4)提速   ≈ {(12.95 - rt) / 12.95 * 100:+.0f}%")


if __name__ == '__main__':
    main()
