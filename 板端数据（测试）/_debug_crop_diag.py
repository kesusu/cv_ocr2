"""
深入诊断: 为什么16个框的Rec输出空字符串?
  
检查维度:
  1. 失败框的crop图片是否正常(尺寸/内容)
  2. crop坐标是否越界或异常
  3. 对比RapidOCR的crop方式是否有差异
"""
import sys, os, time, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    OCRBoardEngine, load_image,
    det_preprocess, det_postprocess,
    get_rotate_crop_image, cls_preprocess, cls_postprocess,
    rec_preprocess, ctc_decode_greedy,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_THRESH, DET_BOX_THRESH,
    DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION,
    CLS_IMAGE_SHAPE, CLS_BATCH_NUM, CLS_THRESH,
    REC_IMAGE_SHAPE, REC_BATCH_NUM, TEXT_SCORE_THRESH,
)

engine = OCRBoardEngine()
engine.init_model()
img = load_image("photos/photo_1.jpg")

# Stage 1-4 完整跑一遍
det_input, resize_info = det_preprocess(
    img, limit_side_len=DET_LIMIT_SIDE_LEN, limit_type=DET_LIMIT_TYPE, pad_to_square=False)
det_output = engine._run_det(det_input)
boxes, box_scores = det_postprocess(
    det_output, img.shape, resize_info,
    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
    use_dilation=DET_USE_DILATION,
)

# Crop所有框
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

# ── 逐个Rec + 详细分析失败case ──
os.makedirs("photos/debug_crops", exist_ok=True)

print(f"\n{'idx':>4} {'rec_sc':>7} {'h':>4} {'w':>4} {'mean_px':>8} {'std_px':>8} {'nonzero%':>8}  text")
print("-" * 100)

drop_indices = []
for i, crop in enumerate(rotated):
    ri = rec_preprocess([crop], tuple(REC_IMAGE_SHAPE[1:]))
    ro = engine._run_rec(ri)
    texts, scores = ctc_decode_greedy(ro, engine.dict_chars)
    
    h, w = crop.shape[:2]
    mean_val = crop.mean()
    std_val = crop.std()
    nonzero_ratio = (crop > 10).sum() / crop.size * 100
    
    t_preview = texts[0][:40].replace('\n', '|')
    
    if scores[0] < TEXT_SCORE_THRESH or not texts[0].strip():
        drop_indices.append(i)
        status = "★ DROP"
        # 保存失败的crop用于目视检查
        cv2.imwrite(f"photos/debug_crops/crop_{i:03d}_sc{scores[0]:.3f}.jpg", crop)
    else:
        status = "ok"
    
    print(f"  {i:>3}  {scores[0]:>7.3f}  {h:>4}  {w:>4}  {mean_val:>8.1f}  {std_val:>8.1f}  {nonzero_ratio:>7.1f}%  [{status}] \"{t_preview}\"")

# ── 统计分析 ──
print(f"\n{'='*70}")
print(f"  DROP indices: {drop_indices}")
print(f"  Total drops: {len(drop_indices)}")
print(f"  Failed crops saved to: photos/debug_crops/")

# 分析失败crop的特征
if drop_indices:
    drop_h = [rotated[i].shape[0] for i in drop_indices]
    drop_w = [rotated[i].shape[1] for i in drop_indices]
    drop_mean = [rotated[i].mean() for i in drop_indices]
    keep_idx = [i for i in range(len(boxes)) if i not in drop_indices]
    
    print(f"\n  --- DROP crops 特征 ---")
    print(f"  height: min={min(drop_h)} max={max(drop_h)} mean={np.mean(drop_h):.0f}")
    print(f"  width:  min={min(drop_w)} max={max(drop_w)} mean={np.mean(drop_w):.0f}")
    print(f"  pixel mean: min={min(drop_mean):.1f} max={max(drop_mean):.1f}")
    
    if keep_idx:
        keep_h = [rotated[i].shape[0] for i in keep_idx]
        keep_w = [rotated[i].shape[1] for i in keep_idx]
        print(f"\n  --- KEEP crops 特征 (对比) ---")
        print(f"  height: min={min(keep_h)} max={max(keep_h)} mean={np.mean(keep_h):.0f}")
        print(f"  width:  min={min(keep_w)} max={max(keep_w)} mean={np.mean(keep_w):.0f}")
        
        # 检查失败的是否都是特别小的crop
        small_drops = sum(1 for i in drop_indices if rotated[i].shape[0] < 20 or rotated[i].shape[1] < 20)
        print(f"\n  ★ 极小crop(h<20或w<20): {small_drops}/{len(drop_indices)}")
