"""
诊断: 逐框比对 Rec 分数 — 找出哪17个框被过滤掉
"""
import sys, time, cv2, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    OCRBoardEngine, load_image,
    det_preprocess, det_postprocess,
    get_rotate_crop_image, cls_preprocess, cls_postprocess,
    rec_preprocess, ctc_decode_greedy,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_THRESH, DET_BOX_THRESH,
    DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION,
    CLS_IMAGE_SHAPE, CLS_BATCH_NUM, CLS_THRESH,
    REC_IMAGE_SHAPE, REC_BATCH_NUM, TEXT_SCORE_THRESH, ENABLE_PREPROCESS, preprocess_image,
)

engine = OCRBoardEngine()
engine.init_model()

img = load_image("photos/photo_1.jpg")
if ENABLE_PREPROCESS:
    img = preprocess_image(img)
print(f"Image: {img.shape}")

# Stage 1: Det
t0 = time.perf_counter()
det_input, resize_info = det_preprocess(
    img, limit_side_len=DET_LIMIT_SIDE_LEN, limit_type=DET_LIMIT_TYPE, pad_to_square=False)
det_output = engine._run_det(det_input)
boxes, box_scores = det_postprocess(
    det_output, img.shape, resize_info,
    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
    use_dilation=DET_USE_DILATION,
)
print(f"\nDet: {len(boxes)} boxes")

# Stage 3: Cls
crops = [get_rotate_crop_image(img, box) for box in boxes]
all_angles, all_cls_scores = [], []
for start in range(0, len(crops), CLS_BATCH_NUM):
    bc = crops[start:start + CLS_BATCH_NUM]
    ci = cls_preprocess(bc, tuple(CLS_IMAGE_SHAPE[1:]))
    co = engine._run_cls(ci)
    angles, cls_sc = cls_postprocess(co, CLS_THRESH)
    all_angles.extend(angles)
    all_cls_scores.extend(cls_sc)

# Rotate if needed
rotated = []
for c, a, s in zip(crops, all_angles, all_cls_scores):
    if a == 1 and s >= CLS_THRESH:
        c = cv2.rotate(c, cv2.ROTATE_180)
    rotated.append(c)

# Stage 4: Rec (逐个打印详情)
print(f"\n{'idx':>4} {'rec_sc':>7} {'box_sc':>7} {'cls_sc':>7} {'status':>9}  text")
print("-" * 100)
all_texts, all_scores = [], []
for i, crop in enumerate(rotated):
    ri = rec_preprocess([crop], tuple(REC_IMAGE_SHAPE[1:]))
    ro = engine._run_rec(ri)
    texts, scores = ctc_decode_greedy(ro, engine.dict_chars)
    all_texts.append(texts[0])
    all_scores.append(scores[0])
    
    status = "KEEP" if scores[0] >= TEXT_SCORE_THRESH else "DROP"
    t_preview = texts[0][:45].replace('\n', '|')
    print(f"  {i:>3}  {scores[0]:>7.3f}  {box_scores[i]:>7.3f}  {all_cls_scores[i]:>7.3f}  {status:>9}  \"{t_preview}\"")

# 统计
sc_arr = np.array(all_scores)
passed = sum(1 for s in all_scores if s >= TEXT_SCORE_THRESH)
empty = sum(1 for t in all_texts if not t.strip())
low_score_idx = [i for i, s in enumerate(all_scores) if s < TEXT_SCORE_THRESH]
print(f"\n{'='*60}")
print(f"  Total: {len(all_texts)}")
print(f"  >={TEXT_SCORE_THRESH}: {passed}  (kept)")
print(f"  <{TEXT_SCORE_THRESH}: {len(low_score_idx)}  (dropped): idx={low_score_idx}")
print(f"  empty strings: {empty}")
print(f"  mean_score={sc_arr.mean():.3f}  min={sc_arr.min():.3f}  max={sc_arr.max():.3f}")
