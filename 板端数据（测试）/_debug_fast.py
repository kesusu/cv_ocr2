#!/usr/bin/env python3
"""快速测试: 单一最佳多图块配置 + 不同NMS参数"""
import sys, os, time, cv2, numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_board import _try_import_sdk, det_postprocess, MODEL_DIR, DET_DLC, load_image

SDK_DET_OUTPUT = 'sigmoid_0.tmp_0'

def run_det(sess, inp):
    r = sess.Execute([SDK_DET_OUTPUT], {'x': inp.astype(np.float32)})
    v = np.array(r.get(SDK_DET_OUTPUT, []))
    if v.ndim == 1:
        s = int(round(v.shape[0]**0.5))
        if s*s == v.shape[0]: v = v.reshape(1,1,s,s)
    return {SDK_DET_OUTPUT: v}

def poly_iou(a, b):
    try:
        p1, p2 = Polygon(a), Polygon(b)
        if not p1.is_valid or not p2.is_valid or not p1.intersects(p2): return 0.0
        i = p1.intersection(p2).area; u = p1.union(p2).area
        return i/u if u > 0 else 0.0
    except: return 0.0

def nms(boxes, scores, iou_t=0.5, min_sc=None):
    n = len(boxes)
    if n == 0: return [], []
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        rest = order[1:]
        sup = {p for p, j in enumerate(rest) if poly_iou(boxes[i], boxes[j]) > iou_t}
        order = [r for p, r in enumerate(rest) if p not in sup]
    rb, rs = [boxes[i] for i in keep], [scores[i] for i in keep]
    if min_sc is not None:
        f = [(b, s) for b, s in zip(rb, rs) if s >= min_sc]
        rb, rs = ([x[0] for x in f], [x[1] for x in f]) if f else ([], [])
    return rb, rs

def main():
    SDK = _try_import_sdk()
    sess = SDK['InferenceSession'](model=os.path.join(MODEL_DIR, DET_DLC),
        platform="qualcomm", framework="snpe", runtime="CPU", log_level="ERROR")
    assert sess.Initialize() == 0
    
    img = load_image('photos/photo_1.jpg')
    h, w = img.shape[:2]
    
    # ===== 只用 tile=640/overlap=0.20/thresh=0.15/box_t=0.30 做一次多图块检测 =====
    print("Running multi-tile detection (tile=640, ov=0.20)...")
    t0 = time.perf_counter()
    
    TILE, OV = 640, 0.20
    THRESH, BT = 0.15, 0.30
    
    nx = max(1, int(np.ceil(w / (TILE * (1-OV))))) if w > TILE else 1
    ny = max(1, int(np.ceil(h / (TILE * (1-OV))))) if h > TILE else 1
    sx = (w-TILE)/max(nx-1,1) if nx>1 else w
    sy = (h-TILE)/max(ny-1,1) if ny>1 else h
    
    raw_b, raw_s = [], []
    for row in range(ny):
        for col in range(nx):
            x1 = min(int(col*sx), max(w-TILE, 0)) if w>TILE else 0
            y1 = min(int(row*sy), max(h-TILE, 0)) if h>TILE else 0
            x2, y2 = min(x1+TILE,w), min(y1+TILE,h)
            actual = img[y1:y2, x1:x2]
            ah, aw = actual.shape[:2]
            
            resized = cv2.resize(actual, (TILE,TILE)) if (ah,aw)!=(TILE,TILE) else actual
            norm = resized.astype(np.float32)/255.0
            norm = (norm - np.array([0.5,0.5,0.5])) / np.array([0.5,0.5,0.5])
            det_in = norm.transpose(2,0,1)[np.newaxis].astype(np.float32)
            
            out = run_det(sess, det_in)
            ri = (TILE, TILE, 1.0, ah, aw)
            tbs, tsc = det_postprocess(out, actual.shape, ri,
                thresh=THRESH, box_thresh=BT, unclip_ratio=1.6,
                max_candidates=500, use_dilation=True)
            
            scx, scy = aw/TILE, ah/TILE
            for box, score in zip(tbs, tsc):
                gb = box.copy()
                gb[:,0]=gb[:,0]*scx+x1; gb[:,1]=gb[:,1]*scy+y1
                gb[:,0]=np.clip(gb[:,0],0,w); gb[:,1]=np.clip(gb[:,1],0,h)
                raw_b.append(gb); raw_s.append(score)
    
    infer_t = time.perf_counter() - t0
    print(f"Multi-tile done: {len(raw_b)} raw boxes in {infer_t:.1f}s ({nx}x{ny}={nx*ny} tiles)")
    
    # ===== 快速测试不同 NMS 参数 =====
    print(f"\n{'Config':<35s} {'Boxes':>6s} {'Score':>7s} {'Gap88':>6s}")
    print("-" * 58)
    
    TARGET = 88
    results = []
    
    for iou_t in [0.3, 0.5, 0.7]:
        for sc_t in [None, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            mb, ms = nms(raw_b, raw_s, iou_t=iou_t, min_sc=sc_t)
            cnt = len(mb)
            avg = float(np.mean(ms)) if ms else 0
            diff = cnt - TARGET
            
            ss = f"{sc_t}" if sc_t is not None else "--"
            tag = " <<<" if 75 <= cnt <= 105 else ""
            print(f"  iou={iou_t:.1f}  score>{ss:>4s}       {cnt:>6d} {avg:>7.3f} {diff:>+5d}{tag}")
            results.append((abs(diff), cnt, avg, f"iou={iou_t:.1f}/sc={ss}", mb, ms))
    
    results.sort(key=lambda x: x[0])
    print(f"\n{'='*60}")
    print(f"TOP-5 closest to target({TARGET}):")
    for rank, (diff, cnt, avg, cfg, mb, ms) in enumerate(results[:5]):
        print(f"  #{rank+1}: {cfg:<25s} -> {cnt} boxes (score={avg:.3f}, gap={cnt-TARGET:+d})")
    
    # 用最佳参数输出详细结果
    best_cfg = results[0]
    _, best_cnt, best_score, best_cfg_str, best_boxes, best_scores = best_cfg
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best_cfg_str}")
    print(f"Result: {best_cnt} boxes, avg_score={best_score:.4f}")

if __name__ == '__main__':
    main()
