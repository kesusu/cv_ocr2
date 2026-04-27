#!/usr/bin/env python3
"""精确找 score 阈值: 目标从83(sc>0.70)追到88"""
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

def rect_nms(boxes, scores, iou_t=0.5, min_sc=None):
    n=len(boxes) 
    if n==0: return [],[]
    rects=np.zeros((n,4))
    for i,b in enumerate(boxes):rects[i]=[b[:,0].min(),b[:,1].min(),b[:,0].max(),b[:,1].max()]
    order=sorted(range(n), key=lambda i: scores[i], reverse=True)
    keep=[]
    areas=(rects[:,2]-rects[:,0])*(rects[:,3]-rects[:,1])
    while len(order)>0:
        i=order[0];keep.append(i)
        if len(order)==1:break
        rest=order[1:]
        if not rest: break
        ix1=np.maximum(rects[i,0],[rects[j,0]for j in rest])
        iy1=np.maximum(rects[i,1],[rects[j,1]for j in rest])
        ix2=np.minimum(rects[i,2],[rects[j,2]for j in rest])
        iy2=np.minimum(rects[i,3],[rects[j,3]for j in rest])
        inter_arr=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou_arr=inter_arr/(areas[i]+np.array([areas[j]for j in rest])-inter_arr)
        order=[rest[j]for j in range(len(rest))if iou_arr[j]<=iou_t]
    rb,rs=[boxes[i]for i in keep],[scores[i]for i in keep]
    if min_sc is not None:
        f=[(b,s)for b,s in zip(rb,rs)if s>=min_sc]
        rb,rs=([x[0]for x in f],[x[1]for x in f])if f else([],[])
    return rb,rs

def main():
    SDK=_try_import_sdk()
    sess=SDK['InferenceSession'](model=os.path.join(MODEL_DIR,DET_DLC),
        platform="qualcomm",framework="snpe",runtime="CPU",log_level="ERROR")
    assert sess.Initialize()==0
    img=load_image('photos/photo_1.jpg')
    h,w=img.shape[:2]; TARGET=88; TILE=640; OV=0.20; TH=0.25; BT=0.40
    
    # 用最佳检测参数做一次多图块检测
    print(f"Multi-tile detection (TILE={TILE},OV={OV},TH={TH},BT={BT})...")
    t0=time.perf_counter()
    
    mean_a=np.array([0.5,0.5,0.5]); std_a=np.array([0.5,0.5,0.5])
    nx=max(1,int(np.ceil(w/(TILE*(1-OV)))))if w>TILE else 1
    ny=max(1,int(np.ceil(h/(TILE*(1-OV)))))if h>TILE else 1
    sx=(w-TILE)/max(nx-1,1)if nx>1 else w; sy=(h-TILE)/max(ny-1,1)if ny>1 else h
    raw_b,raw_s=[],[]
    for row in range(ny):
        for col in range(nx):
            x1=min(int(col*sx),max(w-TILE,0))if w>TILE else 0
            y1=min(int(row*sy),max(h-TILE,0))if h>TILE else 0
            actual=img[y1:min(y1+TILE,h),x1:min(x1+TILE,w)]
            ah,aw=actual.shape[:2]
            resized=cv2.resize(actual,(TILE,TILE))if(ah,aw)!=(TILE,TILE)else actual
            norm=(resized.astype(np.float32)/255.0-mean_a)/std_a
            det_in=norm.transpose(2,0,1)[np.newaxis].astype(np.float32)
            out=run_det(sess,det_in)
            tbs,tsc=det_postprocess(out,actual.shape,(TILE,TILE,1.0,ah,aw),
                thresh=TH,box_thresh=BT,unclip_ratio=1.6,max_candidates=500,use_dilation=True)
            for box,score in zip(tbs,tsc):
                gb=box.copy(); gb[:,0]=gb[:,0]*aw/TILE+x1; gb[:,1]=gb[:,1]*ah/TILE+y1
                raw_b.append(gb); raw_s.append(score)
    
    print(f"Raw: {len(raw_b)} boxes ({time.perf_counter()-t0:.1f}s)")
    
    # 精确扫描 score 阈值 (0.62 ~ 0.74, 步长0.005)
    print(f"\n{'ScoreThresh':>12s} {'Boxes':>6s} {'Score':>7s} {'Gap88'}")
    print("-"*42)
    
    best_diff=9999; best_cfg=None; best_cnt=0; best_score=0
    results=[]
    
    for sc_t in np.arange(0.60, 0.76, 0.005):
        mb,ms=rect_nms(raw_b,raw_s,iou_t=0.5,min_sc=float(sc_t))
        cnt=len(mb); avg=float(np.mean(ms))if ms else 0; diff=cnt-TARGET
        
        marker=""
        if abs(diff)<abs(best_diff):
            best_diff=abs(diff);best_cfg=sc_t;best_cnt=cnt;best_score=avg
            
        if abs(diff)<=15 or 78<=cnt<=100:
            if abs(diff)<=3:marker=" <<<TARGET"
            elif abs(diff)<=7:marker=" ***CLOSE"
            print(f"  {sc_t:>10.3f}   {cnt:>6d} {avg:>7.3f} {diff:+4d}{marker}")
        
        results.append((abs(diff),cnt,avg,float(sc_t)))
    
    print(f"\n{'='*50}")
    print(f"BEST: sc>{best_cfg:.3f} => {best_cnt} boxes(score={best_score:.3f},gap={best_cnt-TARGET:+d})")
    
    # 找到最接近的几个
    results.sort()
    print(f"\nTop-5 closest to {TARGET}:")
    for d,cnt,score,st in results[:5]:
        print(f"  sc>{st:.3f} -> {cnt} boxes (gap={cnt-TARGET:+d})")
    
    # 用最优参数做完整端到端验证 (模拟 ocr_board.py 的流程)
    print(f"\n{'='*50}")
    print(f"END-TO-END VALIDATION with optimal params:")
    mb, ms = rect_nms(raw_b, raw_s, iou_t=0.5, min_score=float(best_cfg))
    final_cnt = len(mb)
    final_avg = float(np.mean(ms)) if ms else 0
    print(f"  Final detection result:")
    print(f"    Boxes: {final_cnt} (target={TARGET}, gap={final_cnt-TARGET:+d})")
    print(f"    Avg Score: {final_avg:.4f}")
    print(f"    Score Threshold: {best_cfg:.3f}")

if __name__=='__main__':
    main()
