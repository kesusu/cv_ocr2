"""
关键诊断: 对比 ocr_board.py vs RapidOCR 的检测框坐标

如果框数相同(都是88)但具体框位置不同,
说明Det后处理有细微差异, 导致某些crop偏移 → Rec失败
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    load_image, det_preprocess, det_postprocess,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_THRESH, DET_BOX_THRESH,
    DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION,
    OCRBoardEngine, MODEL_DIR,
)
import onnxruntime as ort

# ── 方法A: ocr_board 的 Det ──
img = load_image("photos/photo_1.jpg")

det_path = os.path.join(MODEL_DIR,
    [f for f in os.listdir(MODEL_DIR) if 'det' in f.lower() and f.endswith('.onnx')][0])
so = ort.SessionOptions()
so.intra_op_num_threads = 4
session = ort.InferenceSession(det_path, so)

det_input_a, resize_info_a = det_preprocess(
    img, limit_side_len=DET_LIMIT_SIDE_LEN, limit_type=DET_LIMIT_TYPE, pad_to_square=False)
out_name = session.get_outputs()[0].name
output_a = session.run([out_name], {session.get_inputs()[0].name: det_input_a})[0]
boxes_a, scores_a = det_postprocess(
    {out_name: output_a}, img.shape, resize_info_a,
    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
    use_dilation=DET_USE_DILATION,
)
print(f"ocr_board Det: {len(boxes_a)} boxes")

# ── 方法B: RapidOCR 的 Det (直接调RapidOCR内部引擎) ──
from rapidocr import RapidOCR
ocr = RapidOCR(params={
    "Det.thresh": DET_THRESH,
    "Det.box_thresh": DET_BOX_THRESH,
})
result_b = ocr(img)
boxes_b = result_b.boxes if result_b else []
print(f"RapidOCR Det: {len(boxes_b)} boxes")

# ── 框对比分析 ──
print(f"\n{'='*70}")
print(f"  Box 坐标对比 (前10个)")
print(f"{'='*70}")

for i in range(min(10, len(boxes_a), len(boxes_b))):
    ca = np.array(boxes_a[i]).flatten()   # (8,)  4角点xy
    cb = np.array(boxes_b[i]).flatten()
    center_a = ca.reshape(4, 2).mean(axis=0)  # 中心点
    center_b = cb.reshape(4, 2).mean(axis=0)
    dist = np.linalg.norm(center_a - center_b)  # 中心距离
    
    # IoU-like: 用外接矩形重叠度
    ra = [ca[0::2].min(), ca[1::2].min(), ca[0::2].max(), ca[1::2].max()]  # xmin,ymin,xmax,ymax
    rb = [cb[0::2].min(), cb[1::2].min(), cb[0::2].max(), cb[1::2].max()]
    
    # 计算外接矩形IoU
    ix1, iy1 = max(ra[0], rb[0]), max(ra[1], rb[1])
    ix2, iy2 = min(ra[2], rb[2]), min(ra[3], rb[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = (ra[2]-ra[0]) * (ra[3]-ra[1])
    area_b = (rb[2]-rb[0]) * (rb[3]-rb[1])
    union = area_a + area_b - inter
    iou_rect = inter / union if union > 0 else 0
    
    match = "✓" if iou_rect > 0.5 else "?"
    print(f"  [{i:>2}] center_dist={dist:>6.1f}px  rect_IoU={iou_rect:.3f}  {match}")

# ── 更重要: 找出 ocr_board 有但质量差的框 vs RapidOCR对应框 ──
print(f"\n{'='*70}")
print(f"  失败框(index=[2,5,11,17]) 与 RapidOCR 同位框对比")
print(f"{'='*70}")

drop_idx = [2, 5, 11, 17, 31, 39, 64]

# 用中心点匹配法找到RapidOCR中对应的框
def find_best_match(target_box, candidate_boxes):
    """通过中心点距离找最佳匹配"""
    tc = target_box.mean(axis=0)
    best_j, best_dist = -1, 9999
    for j, cb in enumerate(candidate_boxes):
        cc = cb.mean(axis=0)
        d = np.linalg.norm(tc - cc)
        if d < best_dist:
            best_dist = d
            best_j = j
    return best_j, best_dist

for idx in drop_idx:
    if idx >= len(boxes_a):
        continue
    ba = boxes_a[idx]  # ocr_board 的框
    
    # 在RapidOCR中找最接近的匹配框
    matched_j, dist = find_best_match(ba, boxes_b)
    bb = boxes_b[matched_j] if matched_j >= 0 else None
    
    # 计算两个框的面积差异
    def box_area(box):
        pts = np.array(box)
        return cv2.contourArea(pts)
    
    area_a = box_area(ba)
    area_b = box_area(bb) if bb is not None else 0
    
    # crop尺寸
    from ocr_board import get_rotate_crop_image
    crop_a = get_rotate_crop_image(img, ba)
    crop_b = get_rotate_crop_image(img, bb) if bb is not None else None
    
    print(f"\n  ocr_board[{idx}]: h={crop_a.shape[0]} w={crop_a.shape[1]} "
          f"area={area_a:.0f}  mean_px={crop_a.mean():.1f}")
    if bb is not None:
        print(f"  RapidOCR [{matched_j}]: h={crop_b.shape[0]} w={crop_b.shape[1]} "
              f"area={area_b:.0f}  mean_px={crop_b.mean():.1f}")
        
        # 直接用RapidOCR的crop跑我们的Rec看是否也失败
        from ocr_board import rec_preprocess, ctc_decode_greedy, REC_IMAGE_SHAPE
        engine_t = OCRBoardEngine()
        engine_t.init_model()
        
        ri_a = rec_preprocess([crop_a], tuple(REC_IMAGE_SHAPE[1:]))
        ro_a = engine_t._run_rec(ri_a)
        texts_a, scores_a_r = ctc_decode_greedy(ro_a, engine_t.dict_chars)
        
        ri_b = rec_preprocess([crop_b], tuple(REC_IMAGE_SHAPE[1:]))
        ro_b = engine_t._run_rec(ri_b)
        texts_b, scores_b_r = ctc_decode_greedy(ro_b, engine_t.dict_chars)
        
        print(f"  → Rec(ocr_board_crop): \"{texts_a[0][:30]}\" score={scores_a_r[0]:.4f}")
        print(f"  → Rec(RapidOCR_crop):  \"{texts_b[0][:30]}\" score={scores_b_r[0]:.4f}")
        
        if scores_a_r[0] < 0.4 and scores_b_r[0] >= 0.4:
            print(f"  ★★★ 关键发现! 用RapidOCR的同位框crop能成功识别! ★★★")
            print(f"      → 说明问题出在框坐标/形状, 不是Rec模型")
