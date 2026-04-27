#!/usr/bin/env python3
"""v3: 更严格的检测参数 + 更高分数过滤 + 分析框分布"""
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
    n = len(boxes)
    if n == 0: return [], []
    rects = np.zeros((n,4))
    for i,b in enumerate(boxes):
        rects[i] = [b[:,0].min(), b[:,1].min(), b[:,0].max(), b[:,1].max()]
    order = np.argsort(-np.array(scores)).tolist()
    keep = []
    areas = (rects[:,2]-rects[:,0])*(rects[:,3]-rects[:,1])
    while order:
        i = order.pop(0); keep.append(i)
        if not order: break
        rest = np.array(order,dtype=int)
        ix1=np.maximum(rects[i,0],rects[rest,0]); iy1=np.maximum(rects[i,1],rects[rest,1])
        ix2=np.minimum(rects[i,2],rects[rest,2]); iy2=np.minimum(rects[i,3],rects[rest,1])
        inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou=inter/(areas[i]+areas[rest]-inter)
        order=[order[j] for j in range(len(order)) if iou[j]<=iou_t]
    rb,rs=[boxes[i] for i in keep],[scores[i] for i in keep]
    if min_sc is not None:
        f=[(b,s) for b,s in zip(rb,rs) if s>=min_sc]
        rb,rs=([x[0]for x in f],[x[1]for x in f]) if f else([],[])
    return rb,rs

def analyze_boxes(boxes, scores):
    """分析检测框分布"""
    n = len(boxes)
    if n == 0:
        return {}
    
    # 外接矩形尺寸
    widths = []
    heights = []
    areas = []
    centers_x = []
    centers_y = []
    
    for b in boxes:
        xs = b[:, 0]; ys = b[:, 1]
        w = xs.max() - xs.min(); h = ys.max() - ys.min()
        widths.append(w); heights.append(h); areas.append(w*h)
        centers_x.append(xs.mean()); centers_y.append(ys.mean())
    
    # 分数分布
    score_arr = np.array(scores)
    
    return {
        'n': n,
        'width': (np.mean(widths), np.std(widths), min(widths), max(widths)),
        'height': (np.mean(heights), np.std(heights), min(heights), max(heights)),
        'area': (np.mean(areas), np.std(areas)),
        'score_min': score_arr.min(),
        'score_max': score_arr.max(),
        'score_mean': score_arr.mean(),
        'score_median': np.median(score_arr),
        'score_hist': {f">{t}": int((score_arr>t).sum()) for t in [0.3,0.4,0.5,0.6,0.7,0.8]},
        'x_range': (min(centers_x), max(centers_x)),
        'y_range': (min(centers_y), max(centers_y)),
    }

def main():
    SDK = _try_import_sdk()
    sess = SDK['InferenceSession'](model=os.path.join(MODEL_DIR, DET_DLC),
        platform="qualcomm", framework="snpe", runtime="CPU", log_level="ERROR")
    assert sess.Initialize() == 0
    
    img = load_image('photos/photo_1.jpg')
    h, w = img.shape[:2]
    TARGET = 88
    
    print(f"Image: {h}x{w}, Target={TARGET}")
    
    TILE = 640; mean_a = np.array([0.5,0.5,0.5]); std_a = np.array([0.5,0.5,0.5])
    
    # ===== 测试不同的检测参数组合 =====
    test_configs = [
        # (overlap, thresh, box_thresh, desc)
        (0.20, 0.15, 0.30, "base"),
        (0.20, 0.15, 0.45, "strict_bt"),
        (0.20, 0.15, 0.50, "very_strict_bt"),
        (0.20, 0.20, 0.35, "higher_th"),
        (0.20, 0.25, 0.40, "high_th_strict_bt"),
        (0.15, 0.15, 0.30, "less_ov"),
        (0.10, 0.15, 0.30, "minimal_ov"),
        (0.30, 0.15, 0.30, "more_ov"),
        (0.25, 0.15, 0.35, "med_ov_strict"),
        (0.20, 0.12, 0.28, "tuned"),
        (0.18, 0.15, 0.32, "fine_ov_bt"),
    ]
    
    all_results = []
    
    for ov, th, bt, desc in test_configs:
        t0 = time.perf_counter()
        
        nx = max(1, int(np.ceil(w/(TILE*(1-ov))))) if w > TILE else 1
        ny = max(1, int(np.ceil(h/(TILE*(1-ov))))) if h > TILE else 1
        sx = (w-TILE)/max(nx-1,1) if nx>1 else w
        sy = (h-TILE)/max(ny-1,1) if ny>1 else h
        
        raw_b, raw_s = [], []
        for row in range(ny):
            for col in range(nx):
                x1 = min(int(col*sx), max(w-TILE,0)) if w>TILE else 0
                y1 = min(int(row*sy), max(h-TILE,0)) if h>TILE else 0
                actual = img[y1:min(y1+TILE,h), x1:min(x1+TILE,w)]
                ah, aw = actual.shape[:2]
                resized = cv2.resize(actual,(TILE,TILE)) if (ah,aw)!=(TILE,TILE) else actual
                norm = (resized.astype(np.float32)/255.0 - mean_a)/std_a
                det_in = norm.transpose(2,0,1)[np.newaxis].astype(np.float32)
                out = run_det(sess, det_in)
                tbs, tsc = det_postprocess(out, actual.shape, (TILE,TILE,1.0,ah,aw),
                    thresh=th, box_thresh=bt, unclip_ratio=1.6,
                    max_candidates=500, use_dilation=True)
                scx, scy = aw/TILE, ah/TILE
                for box, score in zip(tbs, tsc):
                    gb = box.copy()
                    gb[:,0]=gb[:,0]*scx+x1; gb[:,1]=gb[:,1]*scy+y1
                    raw_b.append(gb); raw_s.append(score)
        
        infer_t = time.perf_counter()-t0
        
        # 分析原始框分布
        info = analyze_boxes(raw_b, raw_s)
        
        # 不同分数过滤的效果
        best_for_this_cfg = None
        best_diff = 9999
        
        for sc_t in [None, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            mb, ms = rect_nms_fast(raw_b, raw_s, iou_t=0.5, min_sc=sc_t)
            cnt = len(mb); avg = float(np.mean(ms)) if ms else 0; diff = abs(cnt-TARGET)
            
            if diff < best_diff:
                best_diff = diff
                best_for_this_cfg = (cnt, avg, diff, sc_t)
            
            ss = f"{sc_t}" if sc_t is not None else "--"
            if abs(diff) <= 30 or cnt >= 60 and cnt <= 120:
                tag = " ***" if abs(diff) <= 15 else ""
                print(f"  [{desc:>16s}] ov={ov:.2f} th={th:.2f} bt={bt:.2f}"
                      f" | raw={len(raw_b):>5d}(mean_s={info['score_mean']:.3f})"
                      f" | sc>{ss:>4s} -> {cnt:>4d} boxes (s={avg:.3f}, gap={cnt-TARGET:+d}){tag}")
        
        all_results.append({
            'desc': desc, 'cfg': f"{desc}/ov={ov:.2f}/th={th:.2f}/bt={bt:.2f}",
            'raw_n': len(raw_b), 
            'best_cnt': best_for_this_cfg[0] if best_for_this_cfg else 0,
            'best_score': best_for_this_cfg[1] if best_for_this_cfg else 0,
            'best_diff': best_for_this_cfg[2] if best_for_this_cfg else 9999,
            'best_sc_t': best_for_this_cfg[3],
            'infer_t': infer_t,
            'info': info,
        })
    
    # 排序输出最佳
    print(f"\n{'='*75}")
    print("RANKED BY BEST RESULT (closest to target):")
    print(f"{'='*75}")
    print(f"{'Rank':>4s} {'Config':<38s} {'Raw':>5s} {'Best':>5s} {'Score':>7s} "
          f"{'Gap':>5s} {'Time':>6s}")
    print("-"*80)
    
    all_results.sort(key=lambda x: x['best_diff'])
    for rank, r in enumerate(all_results[:15]):
        st = r['best_sc_t']
        st_str = f"{st}" if st is not None else "--"
        marker = " <==BEST" if rank==0 else ""
        print(f"  {rank+1:>3d} {r['cfg']:<38s} {r['raw_n']:>5d} {r['best_cnt']:>5d} "
              f"{r['best_score']:>7.3f} {r['best_cnt']-TARGET:>+5d} {r['infer_t']:>5.1f}s{marker}")

if __name__=='__main__':
    main()
