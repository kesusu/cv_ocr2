#!/usr/bin/env python3
"""
攻关 Step 2 (focused): 精确NMS + 质量过滤 — 快速定位最佳配置
基于v1结果: tile=640/overlap=0.2 效果最好(1360原始框), 重点优化NMS和过滤
"""
import sys, os, time, cv2, numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    _try_import_sdk, det_postprocess,
    MODEL_DIR, DET_DLC,
    load_image,
)

SDK_DET_OUTPUT = 'sigmoid_0.tmp_0'

def run_sdk_det(session, det_input):
    input_feed = {'x': det_input.astype(np.float32)}
    result = session.Execute([SDK_DET_OUTPUT], input_feed)
    val = np.array(result.get(SDK_DET_OUTPUT, []))
    if val.ndim == 1:
        total = val.shape[0]
        side = int(round(total ** 0.5))
        if side * side == total:
            val = val.reshape(1, 1, side, side)
    return {SDK_DET_OUTPUT: val}

def preprocess_tile(tile_img, target_size):
    th, tw = tile_img.shape[:2]
    if (th, tw) != (target_size, target_size):
        tile_resized = cv2.resize(tile_img, (target_size, target_size))
    else:
        tile_resized = tile_img
    normalized = tile_resized.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (normalized - mean) / std
    normalized = normalized.transpose((2, 0, 1))
    return np.expand_dims(normalized, axis=0).astype(np.float32), (th, tw)

def poly_iou(p1, p2):
    try:
        a, b = Polygon(p1), Polygon(p2)
        if not a.is_valid or not b.is_valid or not a.intersects(b):
            return 0.0
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def advanced_nms(boxes, scores, iou_t=0.5, min_score=None):
    """多边形精确 IoU NMS (修复索引越界bug)"""
    n = len(boxes)
    if n == 0:
        return [], []
    # 转为 list 避免 numpy 索引问题
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        # 收集要抑制的索引(在 rest 中的位置)
        suppress = set()
        for j_pos, j in enumerate(rest):
            if poly_iou(boxes[i], boxes[j]) > iou_t:
                suppress.add(j_pos)
        # 从 rest 中移除被抑制的
        order = [r for pos, r in enumerate(rest) if pos not in suppress]
    
    rb = [boxes[i] for i in keep]
    rs = [scores[i] for i in keep]
    if min_score is not None:
        filt = [(b, s) for b, s in zip(rb, rs) if s >= min_score]
        rb = [x[0] for x in filt]; rs = [x[1] for x in filt]
    return rb, rs

def multi_tile_detect(session, img, tile_size=640, overlap=0.25,
                      thresh=0.15, box_thresh=0.25):
    h, w = img.shape[:2]
    all_b, all_s = [], []
    nx = max(1, int(np.ceil(w / (tile_size * (1 - overlap))))) if w > tile_size else 1
    ny = max(1, int(np.ceil(h / (tile_size * (1 - overlap))))) if h > tile_size else 1
    sx = (w - tile_size) / max(nx - 1, 1) if nx > 1 else w
    sy = (h - tile_size) / max(ny - 1, 1) if ny > 1 else h
    
    for row in range(ny):
        for col in range(nx):
            x1 = min(int(col * sx), max(w - tile_size, 0)) if w > tile_size else 0
            y1 = min(int(row * sy), max(h - tile_size, 0)) if h > tile_size else 0
            x2, y2 = min(x1 + tile_size, w), min(y1 + tile_size, h)
            actual = img[y1:y2, x1:x2]
            det_input, (act_h, act_w) = preprocess_tile(actual, tile_size)
            output = run_sdk_det(session, det_input)
            ri = (tile_size, tile_size, 1.0, act_h, act_w)
            tbs, tsc = det_postprocess(output, actual.shape, ri,
                thresh=thresh, box_thresh=box_thresh, unclip_ratio=1.6,
                max_candidates=500, use_dilation=True)
            sx_r = act_w / tile_size; sy_r = act_h / tile_size
            for box, score in zip(tbs, tsc):
                gb = box.copy()
                gb[:, 0] = gb[:, 0] * sx_r + x1; gb[:, 1] = gb[:, 1] * sy_r + y1
                gb[:, 0] = np.clip(gb[:, 0], 0, w); gb[:, 1] = np.clip(gb[:, 1], 0, h)
                all_b.append(gb); all_s.append(score)
    return all_b, all_s

def main():
    SDK = _try_import_sdk()
    ISession = SDK['InferenceSession']
    session = ISession(model=os.path.join(MODEL_DIR, DET_DLC),
                       platform="qualcomm", framework="snpe",
                       runtime="CPU", log_level="ERROR")
    assert session.Initialize() == 0
    
    img = load_image('photos/photo_1.jpg')
    oh, ow = img.shape[:2]
    TARGET = 88
    print(f"Image: {oh}x{ow}, Target={TARGET} boxes")
    
    # ===== 基于v1的最佳配置快速测试不同 NMS/过滤 参数 =====
    # v1结论: tile=640/ov=0.2 给出最多有效检测
    
    print("\n--- Focused search on tile=640 ---\n")
    
    configs_to_test = [
        # (overlap, thresh, box_thresh, desc)
        (0.20, 0.15, 0.30, "base"),
        (0.20, 0.10, 0.25, "sensitive"),
        (0.25, 0.15, 0.30, "more_ov"),
        (0.15, 0.15, 0.30, "less_ov"),
        (0.20, 0.15, 0.40, "strict_box"),
        (0.20, 0.10, 0.35, "low_th_strict_bt"),
        (0.30, 0.15, 0.30, "high_ov"),
        (0.20, 0.12, 0.28, "tuned"),
    ]
    
    all_final_results = []
    
    for ov, th, bt, desc in configs_to_test:
        t0 = time.perf_counter()
        raw_b, raw_s = multi_tile_detect(session, img, tile_size=640,
                                          overlap=ov, thresh=th, box_thresh=bt)
        infer_t = time.perf_counter() - t0
        
        # 测试不同的 NMS IoU 和分数阈值组合
        nms_configs = [
            (0.3, None),   # 宽松IoU，不过滤分数
            (0.5, None),
            (0.7, None),   # 严格IoU
            (0.5, 0.30),   # 中等IoU + 分数过滤
            (0.5, 0.40),
            (0.7, 0.30),   # 严格IoU + 分数过滤
            (0.3, 0.50),
        ]
        
        for iou_t, sc_t in nms_configs:
            mb, ms = advanced_nms(raw_b, raw_s, iou_t=iou_t, min_score=sc_t)
            cnt = len(mb)
            avg = float(np.mean(ms)) if ms else 0
            
            diff = abs(cnt - TARGET)
            
            if 60 <= cnt <= 120 or (cnt > 0 and len(all_final_results) < 30):
                marker = " <<<" if 75 <= cnt <= 100 else ""
                st_str = f"{sc_t}" if sc_t else "--"
                total_t = infer_t  # NMS时间忽略不计
                print(f"  [{desc}] ov={ov:.2f} th={th:.2f} bt={bt:.2f}"
                      f" | iou={iou_t:.1f} sc={st_str:>4s}"
                      f" | raw={len(raw_b):>5d} -> {cnt:>4d} boxes"
                      f" score={avg:.3f}{marker}")
                
                all_final_results.append({
                    'cfg': f"{desc}/iou={iou_t:.1f}/sc={st_str}",
                    'cnt': cnt, 'score': avg, 'diff': diff,
                    'time': total_t,
                    'raw_cnt': len(raw_b),
                })
    
    # 排序找最佳
    print("\n" + "=" * 70)
    print("RANKED RESULTS (closest to target=88):")
    print("=" * 70)
    all_final_results.sort(key=lambda x: (x['diff'], -x['score']))
    
    for rank, r in enumerate(all_final_results[:15]):
        gap = r['cnt'] - TARGET
        arrow = "==>" if rank == 0 else "   "
        print(f"  {arrow} #{rank+1}: {r['cfg']:<30s} | "
              f"{r['cnt']:>3d} boxes ({gap:+3d}) | score={r['score']:.3f}")

if __name__ == '__main__':
    main()
