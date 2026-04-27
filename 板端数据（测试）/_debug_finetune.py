#!/usr/bin/env python3
"""精细调参: 在 best config (th=0.25/bt=0.40/ov=0.20/sc>0.7 -> 83框) 附近微调"""
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
    n = len(boxes)
    if n == 0: return [], []
    rects=np.zeros((n,4))
    for i,b in enumerate(boxes):
        rects[i]=[b[:,0].min(),b[:,1].min(),b[:,0].max(),b[:,1].max()]
    order=np.argsort(-np.array(scores)).tolist(); keep=[]
    areas=(rects[:,2]-rects[:,0])*(rects[:,3]-rects[:,1])
    while order:
        i=order.pop(0); keep.append(i)
        if not order: break
        rest=np.array(order,dtype=int)
        ix1=np.maximum(rects[i,0],rects[rest,0]); iy1=np.maximum(rects[i,1],rects[rest,1])
        ix2=np.minimum(rects[i,2],rects[rest,2]); iy2=np.minimum(rects[i,3],rects[rest,1])
        inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou=inter/(areas[i]+areas[rest]-inter)
        order=[order[j] for j in range(len(order)) if iou[j]<=iou_t]
    rb,rs=[boxes[i]for i in keep],[scores[i]for i in keep]
    if min_sc is not None:
        f=[(b,s)for b,s in zip(rb,rs)if s>=min_sc]
        rb,rs=([x[0]for x in f],[x[1]for x in f])if f else([],[])
    return rb,rs

def multi_tile(sess, img, TILE=640, OV=0.20, TH=0.25, BT=0.40):
    h,w=img.shape[:2]; mean_a=np.array([0.5,0.5,0.5]); std_a=np.array([0.5,0.5,0.5])
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
            scx,scy=aw/TILE,ah/TILE
            for box,score in zip(tbs,tsc):
                gb=box.copy(); gb[:,0]=gb[:,0]*scx+x1; gb[:,1]=gb[:,1]*scy+y1
                raw_b.append(gb); raw_s.append(score)
    return raw_b,raw_s

def main():
    SDK=_try_import_sdk()
    sess=SDK['InferenceSession'](model=os.path.join(MODEL_DIR,DET_DLC),
        platform="qualcomm",framework="snpe",runtime="CPU",log_level="ERROR")
    assert sess.Initialize()==0
    img=load_image('photos/photo_1.jpg')
    TARGET=88; TILE=640
    
    print(f"Target={TARGET}, fine-tuning around best config")
    print(f"(th=0.25/bt=0.40/ov=0.20/sc>0.70 => 83 boxes)")
    print(f"\n{'Config':<50s} {'Boxes':>6s} {'Score':>7s} {'Gap'}")
    print("-"*75)
    
    results=[]
    
    # 在最佳配置附近的精细网格搜索
    for ov in [0.18, 0.20, 0.22]:
        for th in [0.22, 0.24, 0.25, 0.26, 0.28]:
            for bt in [0.36, 0.38, 0.40, 0.42, 0.44]:
                t0=time.perf_counter()
                rb,rs=multi_tile(sess,img,TILE=TILE,OV=ov,TH=th,BT=bt)
                it=time.perf_counter()-t0
                
                # 精细的分数阈值扫描
                for sc_t in np.arange(0.62, 0.76, 0.02):
                    mb,ms=rect_nms(rb,rs,iou_t=0.5,min_sc=float(sc_t))
                    cnt=len(mb); avg=float(np.mean(ms))if ms else 0; diff=cnt-TARGET
                    
                    # 只记录接近目标的
                    if abs(diff)<=10 or 80<=cnt<=100:
                        tag=" ***TARGET***"if abs(diff)<=3 else ""
                        cfg=f"ov={ov:.2f}/th={th:.2f}/bt={bt:.2f}/sc>{sc_t:.2f}"
                        print(f"  {cfg:<48s} {cnt:>6d} {avg:>7.3f} {diff:+4d}{tag}")
                        results.append((abs(diff),cnt,avg,cfg,it))
    
    if not results:
        # 如果没有接近目标的，输出所有结果看看分布
        print("\nNo close match found. Running broader scan...")
        for ov in [0.18, 0.20, 0.22]:
            for th in [0.23, 0.25, 0.27]:
                for bt in [0.37, 0.39, 0.41]:
                    t0=time.perf_counter()
                    rb,rs=multi_tile(sess,img,TILE=TILE,OV=ov,TH=th,BT=bt)
                    it=time.perf_counter()-t0
                    for sc_t in [0.65, 0.68, 0.70, 0.72]:
                        mb,ms=rect_nms(rb,rs,min_sc=sc_t)
                        cnt=len(mb); avg=float(np.mean(ms))if ms else 0
                        cfg=f"ov={ov:.2f}/th={th:.2f}/bt={bt:.2f}/sc>{sc_t}"
                        print(f"  {cfg:<48s} {cnt:>6d} {avg:>7.3f} {cnt-TARGET:+4d}")
                        results.append((abs(cnt-TARGET),cnt,avg,cfg,it))
    
    if results:
        results.sort()
        print(f"\n{'='*75}")
        print("TOP-10 CLOSEST TO TARGET:")
        for rank,(ad,cnt,score,cfg,t)in enumerate(results[:10]):
            m="<==BEST"if rank==0 else""
            print(f"  #{rank+1}: {cfg:<45s} -> {cnt} boxes(score={score:.3f},gap={cnt-TARGET:+d}){m}")

if __name__=='__main__':
    main()
