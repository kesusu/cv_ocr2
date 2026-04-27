"""
Rec INT8 vs FP32 原始输出对比（不过滤，直接对比 CTC 解码结果）
"""

import os, sys, time, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'pp-ocrv4_rapid_onnx')
SAVE_DIR = os.path.join(BASE_DIR, 'photos')
DICT_PATH = os.path.join(MODEL_DIR, 'ppocr_keys_v1.txt')

sys.path.insert(0, BASE_DIR)
import onnxruntime as ort
from ocr_board import (
    OCRBoardEngine, load_image,
    REC_IMAGE_SHAPE, REC_BATCH_NUM, DET_LIMIT_SIDE_LEN,
    rec_preprocess, ctc_decode_greedy,
    get_rotate_crop_image, det_preprocess, cls_preprocess,
    det_postprocess, cls_postprocess, sort_boxes_by_layout
)

with open(DICT_PATH, 'r', encoding='utf-8') as f:
    DICT_CHARS = [line.strip() for line in f.readlines()]

REC_H = REC_IMAGE_SHAPE[1]
REC_W = REC_IMAGE_SHAPE[2]

def run_rec_raw(model_file, crops):
    """只跑 Rec 阶段，返回原始 texts+scores（不过滤）"""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, model_file),
        sess_options=opts, providers=['CPUExecutionProvider']
    )
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    all_texts, all_scores = [], []
    t0 = time.time()
    for i in range(0, len(crops), REC_BATCH_NUM):
        batch = crops[i:i+REC_BATCH_NUM]
        batch_data = rec_preprocess(batch, (REC_H, REC_W)).astype(np.float32)
        out = sess.run([out_name], {inp_name: batch_data})[0]
        texts, scores = ctc_decode_greedy(out, DICT_CHARS)
        all_texts.extend(texts)
        all_scores.extend(scores)
    t_elapsed = time.time() - t0
    del sess
    return all_texts, all_scores, t_elapsed


# 准备统一的 Det+Cls 输入框和 crops
print("Loading engine...")
engine = OCRBoardEngine()
engine.init_model()

test_path = os.path.join(SAVE_DIR, 'photo_1.jpg')
img = load_image(test_path)

# Det
det_input, ratio = det_preprocess(img, DET_LIMIT_SIDE_LEN)
h, w = img.shape[:2]
det_out = engine._run_det(det_input)
boxes, box_scores = det_postprocess(det_out, (h,w), ratio)

# Cls
crops_raw = [get_rotate_crop_image(img, b) for b in boxes]
cls_in = cls_preprocess(crops_raw, (48,192))
cls_out = engine._run_cls(cls_in)
angles, _ = cls_postprocess(cls_out, 0.9)  # 低阈值拿全部分类

# 旋转
rotated = []
for crop, angle in zip(crops_raw, angles):
    if angle == 1:
        crop = crop.copy()
        import cv2; crop = cv2.rotate(crop, cv2.ROTATE_180)
    rotated.append(crop)

print(f"Crops prepared: {len(rotated)}\n")

# FP32
print(">>> FP32 Rec")
fp32_txt, fp32_scr, fp32_t = run_rec_raw('ch_PP-OCRv4_rec_mobile.onnx', rotated)
print(f"  Time={fp32_t:.3f}s | N={len(fp32_txt)} | nonempty={sum(1 for t in fp32_txt if t.strip())} | mean={np.mean(fp32_scr):.4f}")

# INT8  
print("\n>>> INT8 Rec")
int8_txt, int8_scr, int8_t = run_rec_raw('ch_PP-OCRv4_rec_mobile_int8.onnx', rotated)
print(f"  Time={int8_t:.3f}s | N={len(int8_txt)} | nonempty={sum(1 for t in int8_txt if t.strip())} | mean={np.mean(int8_scr):.4f}")

# 对比
print("\n" + "=" * 70)
print(f"{'指标':<20} {'FP32':>14} {'INT8':>14} {'差异':>14}")
print("-" * 62)
speedup = fp32_t / int8_t if int8_t > 0 else 0
print(f"{'RecTime(s)':<18} {fp32_t:>14.3f} {int8_t:>14.3f} {speedup:>13.2f}x")
print(f"{'输出数量':<18} {len(fp32_txt):>14} {len(int8_txt):>14} {len(int8_txt)-len(fp32_txt):>+14}")
ne_f = sum(1 for t in fp32_txt if t.strip())
ne_i = sum(1 for t in int8_txt if t.strip())
print(f"{'非空文本数':<18} {ne_f:>14} {ne_i:>14} {ne_i-ne_f:>+14}")
print(f"{'平均置信度':<18} {np.mean(fp32_scr):>14.4f} {np.mean(int8_scr):>14.4f} {np.mean(int8_scr)-np.mean(fp32_scr):>+13.4f}")

# 逐条对比
diff_text, diff_score = [], []
for i in range(max(len(fp32_txt), len(int8_txt))):
    tf = fp32_txt[i] if i < len(fp32_txt) else '(none)'
    sf = fp32_scr[i] if i < len(fp32_scr) else 0
    ti = int8_txt[i] if i < len(int8_txt) else '(none)'
    si = int8_scr[i] if i < len(int8_scr) else 0
    if tf != ti or abs(sf-si) > 0.05:
        diff_text.append(i)

print(f"\n逐条差异: {len(diff_text)}/{max(len(fp32_txt),len(int8_txt))} 条")

if diff_text:
    print("\n--- 前10个差异 ---")
    for idx in diff_text[:10]:
        tf = fp32_txt[idx] if idx < len(fp32_txt) else '?'
        sf = fp32_scr[idx] if idx < len(fp32_scr) else 0
        ti = int8_txt[idx] if idx < len(int8_txt) else '?'
        si = int8_scr[idx] if idx < len(int8_scr) else 0
        print(f"  #{idx+1:2d}: FP32='{tf}'({sf:.3f}) vs INT8='{ti}'({si:.3f})")

# 结论
print("\n" + "=" * 70)
if speedup >= 1.1:
    print(f"★ 加速有效: {speedup:.2f}x")
elif speedup >= 0.98:
    print(f"≈ 速度持平")
else:
    print(f"✗ 反而更慢")

text_match = sum(1 for a,b in zip(fp32_txt,int8_txt) if a==b) if len(fp32_txt)==len(int8_txt) else 0
if text_match >= len(fp32_txt) * 0.95:
    print(f"★ 精度良好: {text_match}/{len(fp32_txt)} 文本一致 ({text_match/len(fp32_txt)*100:.1f}%)")
else:
    print(f"⚠ 精度不足: 仅 {text_match}/{len(fp32_txt)} 一致 ({text_match/max(len(fp32_txt),1)*100:.1f}%)")

recommend = '使用INT8' if speedup >= 1.05 and text_match/ max(len(fp32_txt),1) >= 0.95 else '保持FP32'
print(f"\n建议: {recommend}")
