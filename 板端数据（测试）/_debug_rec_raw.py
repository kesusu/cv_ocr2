"""
诊断: 查看失败crop的原始Rec模型输出
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
)

engine = OCRBoardEngine()
engine.init_model()
img = load_image("photos/photo_1.jpg")

# 获取所有框
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

# Cls
all_angles, all_cls_scores = [], []
for start in range(0, len(crops), CLS_BATCH_NUM):
    bc = crops[start:start + CLS_BATCH_NUM]
    ci = cls_preprocess(bc, tuple(CLS_IMAGE_SHAPE[1:]))
    co = engine._run_cls(ci)
    angles, cls_sc = cls_postprocess(co, CLS_THRESH)
    all_angles.extend(angles)
    all_cls_scores.extend(cls_sc)

rotated = []
for c, a, s in zip(crops, all_angles, all_cls_scores):
    if a == 1 and s >= CLS_THRESH:
        c = cv2.rotate(c, cv2.ROTATE_180)
    rotated.append(c)

# ── 重点分析几个失败案例的原始Rec输出 ──
drop_indices = [2, 5, 11, 17, 31, 39, 64]  # 选几个代表性的
keep_indices = [0, 3, 14, 33, 56]          # 成功案例对比

print("=" * 80)
print("  原始 Rec 模型输出 分析")
print("=" * 80)

for idx in drop_indices + keep_indices:
    crop = rotated[idx]
    ri = rec_preprocess([crop], (48, 320))
    ro_raw = engine._run_rec(ri)   # shape: (1, seq_len, num_classes)
    
    # 详细分析
    probs = ro_raw[0]  # (seq_len, num_classes)
    seq_len, num_classes = probs.shape
    
    pred_idx = np.argmax(probs, axis=1)
    pred_prob = np.max(probs, axis=1)
    
    # 统计非blank的比例
    non_blank_count = (pred_idx != 0).sum()
    total_prob_mean = probs.mean()
    max_prob_overall = probs.max()
    
    # CTC解码
    texts, scores = ctc_decode_greedy(ro_raw, engine.dict_chars)
    
    tag = "★ DROP" if idx in drop_indices else "  OK  "
    
    print(f"\n  [{idx:>3}] {tag}  text=\"{texts[0][:40]}\"  score={scores[0]:.4f}")
    print(f"        raw_out shape=({seq_len}, {num_classes})")
    print(f"        non_blank_tokens={non_blank_count}/{seq_len}  "
          f"max_prob={max_prob_overall:.4f}  mean_prob={total_prob_mean:.4f}")
    
    # 打印top-5预测字符（前10个时间步）
    top_chars = []
    for t in range(min(10, seq_len)):
        pi = pred_idx[t]
        pp = pred_prob[t]
        ch = engine.dict_chars[pi] if pi < len(engine.dict_chars) else f"[{pi}?]"
        top_chars.append(f"{ch}({pp:.2f})")
    print(f"        top10 tokens: {' '.join(top_chars)}")

# ── 额外检查: 失败case是否全部预测为blank? ──
print(f"\n{'='*80}")
print("  全局统计: blank vs 非blank 分布")
for idx in drop_indices:
    crop = rotated[idx]
    ri = rec_preprocess([crop], (48, 320))
    ro = engine._run_rec(ri)
    probs = ro[0]
    pred_idx = np.argmax(probs, axis=1)
    
    blank_ratio = (pred_idx == 0).sum() / len(pred_idx) * 100
    unique_non_blank = set(pred_idx[pred_idx != 0].tolist())
    print(f"  [{idx:>3}] blank%={blank_ratio:.1f}%  "
          f"unique_non_blank_chars={len(unique_non_blank)}  "
          f"indices={sorted(unique_non_blank)[:10]}")
