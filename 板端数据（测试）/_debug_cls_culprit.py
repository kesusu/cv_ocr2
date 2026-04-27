"""
关键假设验证: DLC-GPU Cls 是否对某些框做了错误旋转?
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    OCRBoardEngine, load_image,
    det_preprocess, det_postprocess,
    get_rotate_crop_image, cls_preprocess, cls_postprocess,
    rec_preprocess, ctc_decode_greedy,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_THRESH, DET_BOX_THRESH,
    DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION,
    CLS_IMAGE_SHAPE, CLS_BATCH_NUM, CLS_THRESH,
    REC_IMAGE_SHAPE,
)

engine = OCRBoardEngine()
engine.init_model()
img = load_image("photos/photo_1.jpg")

# 获取88个box + crop
det_input, resize_info = det_preprocess(
    img, limit_side_len=DET_LIMIT_SIDE_LEN, limit_type=DET_LIMIT_TYPE, pad_to_square=False)
det_output = engine._run_det(det_input)
boxes, box_scores = det_postprocess(
    det_output, img.shape, resize_info,
    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
    use_dilation=DET_USE_DILATION,
)
crops = [get_rotate_crop_image(img, box) for box in boxes]

# ── 对每个失败case, 测试三种情况: ──
#   A) 原始crop(不经过Cls) → 直接Rec 
#   B) 经过DLC-Cls后的crop → Rec
#   C) 用ORT-Cls代替 → Rec

drop_idx = [2, 5, 11, 16, 17, 18, 22, 31, 32, 39, 42, 43, 44, 50, 64, 65]

print("=" * 90)
print(f"{'idx':>4} | {'A:无Cls':>12} | {'B:DLC-Cls':>12} | {'diff?':>6} | Cls结果")
print("-" * 90)

for idx in drop_idx:
    crop_raw = crops[idx]  # 原始未旋转的crop
    
    # A) 不经过Cls, 直接Rec
    ri_a = rec_preprocess([crop_raw], tuple(REC_IMAGE_SHAPE[1:]))
    ro_a = engine._run_rec(ri_a)
    texts_a, scores_a = ctc_decode_greedy(ro_a, engine.dict_chars)
    
    # B) 经过DLC Cls
    ci_b = cls_preprocess([crop_raw], tuple(CLS_IMAGE_SHAPE[1:]))
    co_b = engine._run_cls(ci_b)
    angles_b, cls_scores_b = cls_postprocess(co_b, CLS_THRESH)
    
    angle = angles_b[0]
    cls_sc = cls_scores_b[0]
    
    if angle == 1 and cls_sc >= CLS_THRESH:
        crop_rotated = cv2.rotate(crop_raw, cv2.ROTATE_180)
        rotated_tag = f"ROT180(sc={cls_sc:.3f})"
    else:
        crop_rotated = crop_raw.copy()
        rotated_tag = f"NO_ROT(sc={cls_sc:.3f})"
    
    ri_b = rec_preprocess([crop_rotated], tuple(REC_IMAGE_SHAPE[1:]))
    ro_b = engine._run_rec(ri_b)
    texts_b, scores_b = ctc_decode_greedy(ro_b, engine.dict_chars)
    
    # 比较
    diff = "★★★ 不同! ★★★" if texts_a[0] != texts_b[0] or abs(scores_a[0] - scores_b[0]) > 0.1 else ""
    
    print(f"  {idx:>3} | \"{texts_a[0][:10]}\" {scores_a[0]:>.4f} | "
          f"\"{texts_b[0][:10]}\" {scores_b[0]:>.4f} | {diff:>6} | {rotated_tag}")

# ── 全局统计: 有多少被Cls误旋转了 ──
print(f"\n{'='*90}")
print("  全局Clas分析 (全部88个框)")
print(f"{'='*90}")

ci_all = cls_preprocess(crops, tuple(CLS_IMAGE_SHAPE[1:]))
co_all = engine._run_cls(ci_all)
angles_all, cls_scores_all = cls_postprocess(co_all, CLS_THRESH)

rotated_count = sum(1 for a, s in zip(angles_all, cls_scores_all) if a == 1 and s >= CLS_THRESH)
print(f"  被Cls旋转180°的框数: {rotated_count}/88")

# 打印所有被旋转的框的详情
if rotated_count > 0:
    print(f"\n  被旋转的框列表:")
    for i, (a, s) in enumerate(zip(angles_all, cls_scores_all)):
        if a == 1 and s >= CLS_THRESH:
            h, w = crops[i].shape[:2]
            print(f"    [{i:>3}] cls_score={s:.3f}  crop_size=({h}x{w})  box_score={box_scores[i]:.3f}")
