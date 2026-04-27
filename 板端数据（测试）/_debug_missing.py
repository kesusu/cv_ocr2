"""
精确诊断: 每个RapidOCR结果在 ocr_board 流程中哪一步丢失的
"""
import sys
sys.path.insert(0, '/home/fibo/cv')
import cv2
import numpy as np
from rapidocr import RapidOCR
from ocr_board import OCRBoardEngine, det_preprocess, det_postprocess, \
    get_rotate_crop_image, rec_preprocess, cls_preprocess, ctc_decode_greedy, \
    DET_THRESH, DET_BOX_THRESH, DET_UNCLIP_RATIO, DET_MAX_CANDIDATES, DET_USE_DILATION, \
    TEXT_SCORE_THRESH, CLS_IMAGE_SHAPE, REC_IMAGE_SHAPE, CLS_BATCH_NUM, REC_BATCH_NUM, CLS_THRESH

# ============================================================
# Part 1: RapidOCR 参考
# ============================================================
print("Loading RapidOCR...")
rapid = RapidOCR()
output = rapid('photos/photo_1.jpg')
rapid_results = []
for i in range(len(output.boxes)):
    rapid_results.append((output.boxes[i].copy(), output.txts[i], float(output.scores[i])))
print(f"RapidOCR: {len(rapid_results)} results")

# ============================================================
# Part 2: Board 引擎 - 分步执行
# ============================================================
print("\nLoading Board Engine...")
engine = OCRBoardEngine()
engine.init_model()

img = cv2.imdecode(np.fromfile('photos/photo_1.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)

# Step 1: Det
print("\n--- Step 1: Detection ---")
det_input, resize_info = det_preprocess(img, 736, 'min')
det_output = engine._run_det(det_input)
raw_boxes, raw_scores = det_postprocess(
    det_output, img.shape, resize_info,
    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
    use_dilation=DET_USE_DILATION,
)
print(f"Det postprocess: {len(raw_boxes)} boxes")

# Step 2: 对每个 RapidOCR 结果，找最近匹配的 board box
print("\n--- Step 2: Match Analysis ---")
def box_center(b):
    return b.mean(axis=0)

def box_iou(a, b):
    """简化: 用中心点距离判断是否同一框"""
    ca, cb = box_center(a), box_center(b)
    dist = np.linalg.norm(ca - cb)
    # 用框的平均尺寸归一化
    avg_size = (np.linalg.norm(a[0]-a[1]) + np.linalg.norm(a[0]-a[3])) / 2
    return dist < avg_size * 0.5  # 中心距小于半平均尺寸则视为匹配

# 对每个 RapidOCR 结果，检查它在 board 流程中的状态
for ri, (rbox, rtext, rscore) in enumerate(rapid_results):
    # 跳过水印
    if any(kw in rtext for kw in ['MJPG', 'fps', 'CPU:', 'RAM:', 'App:', 'Photos:', 'SPACE:', 'shot']):
        continue
    
    # 找最近的 board det box
    best_match_idx = -1
    best_dist = float('inf')
    for bi, bbox in enumerate(raw_boxes):
        c1, c2 = box_center(rbox), box_center(bbox)
        d = np.linalg.norm(c1 - c2)
        if d < best_dist:
            best_dist = d
            best_match_idx = bi
    
    rc = box_center(rbox)
    matched = best_match_idx >= 0 and best_dist < 50  # 50px 阈值
    
    if not matched:
        print(f"[NO MATCH #{ri}] '{rtext}' score={rscore:.3f} @ ({rc[0]:.0f},{rc[1]:.0f}) "
              f"— nearest dist={best_dist:.1f}px")
        continue
    
    # 这个 RapidOCR 框在 board 中有对应的 det box
    # 现在检查它是否通过了最终过滤
    bbox = raw_boxes[best_match_idx]
    bscore_raw = raw_scores[best_match_idx]
    
    # 在最终结果中搜索是否有匹配文本
    bresult = engine.recognize('photos/photo_1.jpg')
    found_in_final = False
    final_text = ''
    final_score = 0
    for bt, bs, bb in zip(bresult['texts'], bresult['scores'], bresult['boxes']):
        bc = bb.mean(axis=0)
        d2 = np.linalg.norm(rc - bc)
        if d2 < 50:
            found_in_final = True
            final_text = bt
            final_score = bs
            break
    
    status = "✓ IN FINAL" if found_in_final else "✗ MISSING FROM FINAL"
    
    if rtext != final_text or abs(rscore - final_score) > 0.1:
        print(f"[{status} #{ri}] rapid='{rtext}'({rscore:.3f}) board='{final_text}'({final_score:.3f})")
    elif not found_in_final:
        print(f"[{status} #{ri}] '{rtext}'({rscore:.3f}) — was in det but lost before final output")
