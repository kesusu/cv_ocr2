#!/usr/bin/env python3
"""
Cls GPU 恢复实验: 扫描不同 CLS_THRESH 值下 DLC-GPU Cls 的效果
目标:找到一个阈值, 让误旋转数≈0 同时保留GPU加速收益
"""
import sys, os, time, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    OCRBoardEngine, load_image,
    det_preprocess, det_postprocess,
    get_rotate_crop_image, cls_preprocess,
    rec_preprocess, ctc_decode_greedy,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_THRESH, DET_BOX_THRESH,
    DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION,
    CLS_IMAGE_SHAPE, CLS_BATCH_NUM, REC_BATCH_NUM, REC_IMAGE_SHAPE,
    MODEL_DIR, CLS_DLC, TEXT_SCORE_THRESH,
    _try_import_sdk,
)

SDK_CLS_OUTPUT = 'save_infer_model/scale_0.tmp_1'


def run_dlc_cls_batch(session, input_batch):
    """DLC-Cls批量推理(SDK逐样本执行)"""
    results = []
    for i in range(input_batch.shape[0]):
        single = input_batch[i:i+1]
        result = session.Execute([SDK_CLS_OUTPUT], {'x': single.astype(np.float32)})
        val = np.array(result.get(SDK_CLS_OUTPUT, []))
        results.append(val)
    combined = np.concatenate(results, axis=0)
    return combined


def cls_postprocess_raw(preds_array):
    """兼容 DLC/ORT 两种输出格式的后处理"""
    arr = np.array(preds_array)
    # 调试: 打印shape
    print(f"    [DEBUG] preds shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}")

    if arr.ndim == 1:
        # 可能是单样本输出 (2,) 或展平的batch
        if arr.size == 2:
            arr = arr.reshape(1, 2)
        else:
            # 展平的多样本? 尝试恢复
            n = arr.size // 2
            if n * 2 == arr.size:
                arr = arr.reshape(n, 2)

    if arr.ndim == 2:
        angles = []
        scores = []
        for pred in arr:
            idx = int(np.argmax(pred))
            score = float(pred[idx])
            angles.append(idx)
            scores.append(score)
        return angles, scores
    elif arr.ndim == 3 and arr.shape[0] == 1:
        return cls_postprocess_raw(arr[0])
    else:
        print(f"    [WARN] Unexpected shape: {arr.shape}, trying flatten...")
        flat = arr.flatten()
        return cls_postprocess_raw(flat)


def main():
    print("=" * 80)
    print("  Cls DLC-GPU 恢复实验: CLS_THRESH 阈值扫描")
    print("=" * 80)

    SDK = _try_import_sdk()
    if SDK is None:
        print("ERROR: SDK not available!")
        return None

    ISession = SDK['InferenceSession']
    cls_dlc_path = os.path.join(MODEL_DIR, CLS_DLC)
    cls_session = ISession(
        model=cls_dlc_path, platform="qualcomm", framework="snpe",
        runtime="GPU", log_level="ERROR", profile_level=5,
    )
    ret = cls_session.Initialize()
    if ret != 0:
        print(f"ERROR: DLC-Cls GPU init failed: ret={ret}")
        return None
    print(f"[OK] DLC-Cls GPU loaded!")

    engine = OCRBoardEngine()
    engine.init_model()

    img = load_image('photos/photo_1.jpg')
    ih, iw = img.shape[:2]
    print(f"\n  Image: {ih}x{iw}")

    # ── Det ──
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
    total = len(boxes)
    print(f"  Det: {total} boxes ({det_t:.3f}s)")

    # ── DLC-GPU Cls 原始输出 ──
    ci_all = cls_preprocess(crops, tuple(CLS_IMAGE_SHAPE[1:]))
    print(f"  Cls input: {ci_all.shape}")

    t_cls = time.perf_counter()
    raw_pred_arr = run_dlc_cls_batch(cls_session, ci_all)
    cls_t = time.perf_counter() - t_cls

    print(f"  Cls DLC output raw: shape={raw_pred_arr.shape}, time={cls_t:.4f}s")

    raw_angles, raw_scores = cls_postprocess_raw(raw_pred_arr)

    dlc_rot_idx = [i for i, a in enumerate(raw_angles) if a == 1]
    dlc_rot_sc = [raw_scores[i] for i in dlc_rot_idx]

    print(f"\n  DLC判定ROT180: {len(dlc_rot_idx)}/{total} 框")
    if dlc_rot_sc:
        sorted_sc = sorted(dlc_rot_sc)
        print(f"  置信度范围: [{min(dlc_rot_sc):.4f}, {max(dlc_rot_sc):.4f}]")
        print(f"  置信度分布: {[round(s, 4) for s in sorted_sc]}")

    # ── 阈值扫描 ──
    print(f"\n{'='*100}")
    print(f"  {'CLS_THRESH':>12} | {'实际旋转':>8} | {'Rec成功':>8} | {'AvgScore':>9} | "
          f"{'Texts':>6} | {'Gap':>5} | {'状态'}")
    print("-" * 100)

    best_thresh = None
    best_result = None

    test_thresholds = [0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999, 0.9995, 0.9999]

    for thresh in test_thresholds:
        rotated_crops = []
        actual_rotated = 0
        for i, crop in enumerate(crops):
            if raw_angles[i] == 1 and raw_scores[i] >= thresh:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
                actual_rotated += 1
            rotated_crops.append(crop)

        t_rec = time.perf_counter()
        all_texts, all_scores = [], []
        for start in range(0, len(rotated_crops), REC_BATCH_NUM):
            ri = rec_preprocess(rotated_crops[start:start+REC_BATCH_NUM], tuple(REC_IMAGE_SHAPE[1:]))
            ro = engine._run_rec(ri)
            texts, scores = ctc_decode_greedy(ro, engine.dict_chars)
            all_texts.extend(texts)
            all_scores.extend(scores)
        rec_t = time.perf_counter() - t_rec

        valid = [(t, s) for t, s in zip(all_texts, all_scores) if s >= TEXT_SCORE_THRESH and t.strip()]
        n_valid = len(valid)
        avg_sc = float(np.mean([s for _, s in valid])) if valid else 0
        gap = n_valid - 88

        if gap == 0:
            status = "★★★ 完美!"
        elif abs(gap) <= 2:
            status = "✓✓ 可接受"
        elif gap < -3:
            status = f"✗ 缺{-gap}个"
        else:
            status = f"? 多余{gap}个"

        tag = ""
        is_better = (best_result is None or
                     abs(gap) < abs(best_result['gap']) or
                     (abs(gap) == abs(best_result['gap']) and avg_sc > best_result.get('score', 0)))
        if is_better:
            tag = " ★★ BEST"

        print(f"  {thresh:>11.4f} | {actual_rotated:>8d} | {n_valid:>8d} | {avg_sc:>9.4f} | "
              f"{n_valid:>6d} | {gap:>+5d} | {status}{tag}")

        if tag:
            best_thresh = thresh
            best_result = {
                'thresh': thresh, 'rotated': actual_rotated,
                'valid': n_valid, 'score': avg_sc, 'gap': gap,
                'cls_t': cls_t, 'rec_t': rec_t,
            }

    # ── 最终结论 ──
    print(f"\n{'='*100}")
    print(f"  ★ 最佳配置:")
    print(f"     CLS_THRESH = {best_thresh}")
    print(f"     Texts      = {best_result['valid']} (目标88, 差异={best_result['gap']:+d})")
    print(f"     AvgScore   = {best_result['score']:.4f}")
    print(f"     实际旋转数 = {best_result['rotated']}")
    print(f"     Cls耗时    = {cls_t:.4f}s (ORT-CPU基准 ~0.31s, 加速比={0.31/max(cls_t,0.001):.1f}x)")

    # ── 如果达到88框，展示完整验证 ──
    if best_result and best_result['valid'] == 88:
        print(f"\n{'='*100}")
        print(f"  ★ CLS_THRESH={best_thresh} 下逐框验证 (仅显示异常):")

        rotated_final = []
        for i, crop in enumerate(crops):
            if raw_angles[i] == 1 and raw_scores[i] >= best_thresh:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
            rotated_final.append(crop)

        all_texts, all_scores = [], []
        for start in range(0, len(rotated_final), REC_BATCH_NUM):
            ri = rec_preprocess(rotated_final[start:start+REC_BATCH_NUM], tuple(REC_IMAGE_SHAPE[1:]))
            ro = engine._run_rec(ri)
            texts, scores = ctc_decode_greedy(ro, engine.dict_chars)
            all_texts.extend(texts)
            all_scores.extend(scores)

        errors = []
        for i in range(total):
            text = all_texts[i][:45]
            sc = all_scores[i]
            if sc < TEXT_SCORE_THRESH or not text.strip():
                is_dlc_rot = (raw_angles[i] == 1)
                actually_rot = (is_dlc_rot and raw_scores[i] >= best_thresh)
                errors.append((i, raw_angles[i], raw_scores[i], actually_rot, text, sc))

        if errors:
            print(f"  ✗ 失败框({len(errors)}个):")
            for i, angle, dsc, rot, txt, sc in errors:
                atag = f"DLC=ROT180({dsc:.4f})→实际={'ROT180' if rot else 'keep'}" if angle == 1 else f"DLC=keep({dsc:.4f})"
                print(f"    [{i:>3}] {atag} → \"{txt}\" score={sc:.3f}")
        else:
            print(f"  ✓✓✓ 全部{total}个框识别成功! 无一失败!")

    cls_session.Destroy()
    return best_thresh, best_result


if __name__ == '__main__':
    main()
