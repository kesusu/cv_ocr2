#!/usr/bin/env python3
"""快速版: 矩形IoU粗筛 + 精确参数搜索"""
import sys, os, time, cv2, numpy as np
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

def rect_nms_fast(boxes, scores, iou_t=0.5, min_sc=None):
    """超快矩形NMS (基于外接矩形) - O(N log N)"""
    n = len(boxes)
    if n == 0: return [], []
    # 预计算每个框的外接矩形 [x1,y1,x2,y2]
    rects = np.zeros((n, 4))
    for i, b in enumerate(boxes):
        rects[i] = [b[:,0].min(), b[:,1].min(), b[:,0].max(), b[:,1].max()]
    
    order = np.argsort(-np.array(scores)).tolist()
    keep = []
    areas = (rects[:,2] - rects[:,0]) * (rects[:,3] - rects[:,1])
    
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        
        rest = np.array(order, dtype=int)
        ix1 = np.maximum(rects[i,0], rects[rest,0])
        iy1 = np.maximum(rects[i,1], rects[rest,1])
        ix2 = np.minimum(rects[i,2], rects[rest,2])
        iy2 = np.minimum(rects[i,3], rects[rest,1])
        
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        iou = inter / (areas[i] + areas[rest] - inter)
        
        order = [order[j] for j in range(len(order)) if iou[j] <= iou_t]
    
    rb, rs = [boxes[i] for i in keep], [scores[i] for i in keep]
    if min_sc is not None:
        f = [(b,s) for b,s in zip(rb,rs) if s >= min_sc]
        rb, rs = ([x[0] for x in f], [x[1] for x in f]) if f else ([],[])
    return rb, rs

def main():
    SDK = _try_import_sdk()
    sess = SDK['InferenceSession'](model=os.path.join(MODEL_DIR, DET_DLC),
        platform="qualcomm", framework="snpe", runtime="CPU", log_level="ERROR")
    assert sess.Initialize() == 0
    
    img = load_image('photos/photo_1.jpg')
    h, w = img.shape[:2]
    TARGET = 88
    
    print(f"Image: {h}x{w}, Target={TARGET}")
    
    # ===== Step 1: 多图块检测 =====
    print("\n--- Multi-tile detection ---")
    t0 = time.perf_counter()
    TILE, OV, THRESH, BT = 640, 0.20, 0.15, 0.30
    
    nx = max(1, int(np.ceil(w/(TILE*(1-OV))))) if w > TILE else 1
    ny = max(1, int(np.ceil(h/(TILE*(1-OV))))) if h > TILE else 1
    sx = (w-TILE)/max(nx-1,1) if nx>1 else w
    sy = (h-TILE)/max(ny-1,1) if ny>1 else h
    
    raw_b, raw_s = [], []
    mean_arr = np.array([0.5,0.5,0.5]); std_arr = np.array([0.5,0.5,0.5])
    
    for row in range(ny):
        for col in range(nx):
            x1 = min(int(col*sx), max(w-TILE,0)) if w>TILE else 0
            y1 = min(int(row*sy), max(h-TILE,0)) if h>TILE else 0
            actual = img[y1:min(y1+TILE,h), x1:min(x1+TILE,w)]
            ah, aw = actual.shape[:2]
            resized = cv2.resize(actual,(TILE,TILE)) if (ah,aw)!=(TILE,TILE) else actual
            norm = (resized.astype(np.float32)/255.0 - mean_arr)/std_arr
            det_in = norm.transpose(2,0,1)[np.newaxis].astype(np.float32)
            
            out = run_det(sess, det_in)
            tbs, tsc = det_postprocess(out, actual.shape, (TILE,TILE,1.0,ah,aw),
                thresh=THRESH, box_thresh=BT, unclip_ratio=1.6,
                max_candidates=500, use_dilation=True)
            
            scx, scy = aw/TILE, ah/TILE
            for box, score in zip(tbs, tsc):
                gb = box.copy()
                gb[:,0]=gb[:,0]*scx+x1; gb[:,1]=gb[:,1]*scy+y1
                raw_b.append(gb); raw_s.append(score)
    
    infer_t = time.perf_counter()-t0
    print(f"Raw: {len(raw_b)} boxes from {nx}x{ny}={nx*ny} tiles ({infer_t:.1f}s)")
    
    # ===== Step 2: 快速NMS参数网格 =====
    print(f"\n{'Config':<35s} {'Boxes':>6s} {'Score':>7s} {'Gap':>5s} {'Time'}")
    print("-"*65)
    
    results = []
    
    for iou_t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for sc_t in [None, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            t1 = time.perf_counter()
            mb, ms = rect_nms_fast(raw_b, raw_s, iou_t=iou_t, min_sc=sc_t)
            nt = time.perf_counter()-t1
            
            cnt = len(mb); avg = float(np.mean(ms)) if ms else 0; diff = cnt-TARGET
            ss = f"{sc_t}" if sc_t is not None else "--"
            tag = " <<<" if 75<=cnt<=105 else ""
            print(f"  iou={iou_t:.1f} sc>{ss:>4s}       {cnt:>6d} {avg:>7.3f} {diff:>+4d} {nt:.2f}s{tag}")
            results.append((abs(diff), cnt, avg, diff,
                           f"iou={iou_t:.1f}/sc={ss}", nt, len(ms)>0))
    
    results.sort()
    print(f"\n{'='*65}")
    print(f"TOP-10 closest to target({TARGET}):")
    print(f"{'#':>2s} {'Config':<28s} {'Boxes':>6s} {'Score':>7s} {'Gap':>5s}")
    print("-"*55)
    for rank, (adiff, cnt, avg, diff, cfg, _, _) in enumerate(results[:10]):
        marker = " <==BEST" if rank==0 else ""
        print(f"  {rank+1:>2d} {cfg:<28s} {cnt:>6d} {avg:>7.3f} {diff:>+4d}{marker}")
    
    best = results[0]
    print(f"\nBEST: {best[4]} -> {best[1]} boxes (score={best[2]:.3f}, gap={best[3]:+d})")

if __name__=='__main__':
    main()
