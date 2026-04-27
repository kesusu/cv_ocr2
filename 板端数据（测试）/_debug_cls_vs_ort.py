"""
验证: ORT-Cls 是否也误旋转这20个框?
如果是 → 说明是图片本身的问题
如果否 → 纯粹DLC-Cls模型问题 → 解决方案: Cls改回ORT或提高阈值
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
    REC_IMAGE_SHAPE, MODEL_DIR,
)
import onnxruntime as ort

# ── 加载 ORT-Cls 模型作为对照 ──
cls_onnx_path = os.path.join(MODEL_DIR,
    [f for f in os.listdir(MODEL_DIR) if 'cls' in f.lower() and f.endswith('.onnx')][0])
so = ort.SessionOptions()
so.intra_op_num_threads = 4
ort_cls_session = ort.InferenceSession(cls_onnx_path, so)
print(f"[OK] ORT-Cls loaded: {os.path.basename(cls_onnx_path)}")

# ── 加载完整引擎(用DLC-Clas) ──
engine = OCRBoardEngine()
engine.init_model()
img = load_image("photos/photo_1.jpg")

# Det
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

# ── 对比DLC-Clas vs ORT-Clas ──
ci_all = cls_preprocess(crops, tuple(CLS_IMAGE_SHAPE[1:]))

# DLC-Clas 结果
co_dlc = engine._run_cls(ci_all)
angles_dlc, scores_dlc = cls_postprocess(co_dlc, CLS_THRESH)

# ORT-Clas 结果
ort_in_name = ort_cls_session.get_inputs()[0].name
ort_out_name = ort_cls_session.get_outputs()[0].name
co_ort_raw = ort_cls_session.run([ort_out_name], {ort_in_name: ci_all.astype(np.float32)})[0]
angles_ort, scores_ort = cls_postprocess({ort_out_name: co_ort_raw}, CLS_THRESH)

# ── 对比表格 ──
print(f"\n{'='*95}")
print(f"{'idx':>4} | {'DLC-Cls':>15} | {'ORT-Cls':>15} | {'一致?':>6} | 无Cls时Rec结果")
print("-" * 95)

disagree_count = 0
dlc_only_rotate = 0

for i in range(len(boxes)):
    dlc_angle = angles_dlc[i]
    dlc_sc = scores_dlc[i]
    ort_angle = angles_ort[i]
    ort_sc = scores_ort[i]
    
    dlc_tag = f"ROT180({dlc_sc:.3f})" if dlc_angle == 1 and dlc_sc >= CLS_THRESH else f"keep({dlc_sc:.3f})"
    ort_tag = f"ROT180({ort_sc:.3f})" if ort_angle == 1 and ort_sc >= CLS_THRESH else f"keep({ort_sc:.3f})"
    
    # 判断是否一致
    dlc_rot = (dlc_angle == 1 and dlc_sc >= CLS_THRESH)
    ort_rot = (ort_angle == 1 and ort_sc >= CLS_THRESH)
    agree = "✓" if dlc_rot == ort_rot else "✗ ✗ ✗"
    
    if dlc_rot != ort_rot:
        disagree_count += 1
        if dlc_rot and not ort_rot:
            dlc_only_rotate += 1
    
    # 无Cls时的Rec结果
    crop_raw = crops[i]
    ri = rec_preprocess([crop_raw], tuple(REC_IMAGE_SHAPE[1:]))
    ro = engine._run_rec(ri)
    texts, sc = ctc_decode_greedy(ro, engine.dict_chars)
    
    row_mark = ""
    if dlc_rot != ort_rot:
        row_mark = " ← ★ DLC独有旋转!"
    
    print(f"  {i:>3} | {dlc_tag:>15} | {ort_tag:>15} | {agree:>6} | \"{texts[0][:25]}\"{row_mark}")

print(f"\n{'='*95}")
print(f"  统计:")
print(f"    总框数: {len(boxes)}")
print(f"    Cls判断不一致: {disagree_count}")
print(f"    DLC独有旋转(DLC说转, ORT说不转): {dlc_only_rotate}")
print(f"    ORT独有旋转(ORT说转, DLC说不转): {disagree_count - dlc_only_rotate}")

# ── 结论与修复建议 ──
print(f"\n{'='*95}")
print(f"  修复建议:")
if dlc_only_rotate > 0:
    print(f"  ★ DLC-Cls 有 {dlc_only_rotate} 个误旋转! 方案:")
    print(f"    1. [推荐] USE_DLC_CLS=False → 改回ORT-Cls (准确但慢)")
    print(f"    2. 提高 CLS_THRESH 从{CLS_THRESH}到0.99+ (减少误旋转)")
    print(f"    3. 只在DLC-Cls置信度极高(>0.999)时才信任")
else:
    print(f"  DLC和ORT基本一致, 问题可能在其他地方")
