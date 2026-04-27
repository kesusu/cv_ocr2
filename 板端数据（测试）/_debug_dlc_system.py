#!/usr/bin/env python3
"""
系统性诊断: 找到 DLC Det 在 SDK-CPU 下的最佳参数配置
目标: 让 DLC Det 检测框数量接近 ORT 的 88 框
"""
import sys, os, time, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    _try_import_sdk, det_preprocess, det_postprocess,
    MODEL_DIR, DET_DLC,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE,
    DET_THRESH, DET_BOX_THRESH, DET_UNCLIP_RATIO,
    DET_MAX_CANDIDATES, DET_USE_DILATION,
    load_image,
)

SDK_DET_OUTPUT = 'sigmoid_0.tmp_0'

def run_sdk_det(session, det_input):
    """用正确的方式调用 SDK Det 推理"""
    input_feed = {'x': det_input.astype(np.float32)}
    result = session.Execute([SDK_DET_OUTPUT], input_feed)
    val = np.array(result.get(SDK_DET_OUTPUT, []))
    if val.ndim == 1:
        total = val.shape[0]
        side = int(round(total ** 0.5))
        if side * side == total:
            val = val.reshape(1, 1, side, side)
    return {SDK_DET_OUTPUT: val}

def main():
    SDK = _try_import_sdk()
    if SDK is None:
        print("ERROR: SDK not available")
        return
    
    det_dlc_path = os.path.join(MODEL_DIR, DET_DLC)
    ISession = SDK['InferenceSession']
    
    img = load_image('photos/photo_1.jpg')
    orig_h, orig_w = img.shape[:2]
    print(f"Original image: {orig_h}x{orig_w}")
    print("=" * 70)
    
    all_results = {}
    
    # ===== 1. ORT 基准 (参考) =====
    print("\n[1/4] ORT Baseline (reference)...")
    import onnxruntime as ort
    onnx_path = os.path.join(MODEL_DIR, DET_DLC.replace('.dlc', '.onnx'))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    ort_session = ort.InferenceSession(onnx_path, so)
    
    t0 = time.perf_counter()
    det_input, resize_info = det_preprocess(img, DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, pad_to_square=False)
    outputs = ort_session.run(None, {'x': det_input})
    infer_t = time.perf_counter() - t0
    
    arr = outputs[0]
    pm = arr[0, 0]
    print(f"  Output shape={arr.shape}, max={arr.max():.6f}, mean={arr.mean():.6f}")
    print(f"  ProbMap: {pm.shape}, pixels>0.3={(pm>0.3).sum()}, >0.1={(pm>0.1).sum()}")
    
    ort_output = {'output': arr}
    boxes, scores = det_postprocess(
        ort_output, img.shape, resize_info,
        thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
        unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
        use_dilation=DET_USE_DILATION,
    )
    all_results['ORT-ref'] = (len(boxes), float(np.mean(scores)) if scores else 0)
    print(f"  => {len(boxes)} boxes, avg_score={all_results['ORT-ref'][1]:.4f}, time={infer_t:.3f}s")
    del ort_session
    
    # ===== 2. SDK-CPU + ORT 预处理 (大尺寸输入) =====
    for runtime in ["CPU", "GPU"]:
        name = f"SDK-{runtime}"
        print(f"\n[{'2' if runtime=='CPU' else '3'}/4] {name} + ORT-preprocess (large input)...")
        
        try:
            session = ISession(
                model=det_dlc_path,
                platform="qualcomm",
                framework="snpe",
                runtime=runtime,
                log_level="ERROR",
            )
            ret = session.Initialize()
            if ret != 0:
                print(f"  [FAIL] Initialize ret={ret}")
                continue
            
            t0 = time.perf_counter()
            det_input, resize_info = det_preprocess(img, DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, pad_to_square=False)
            
            output = run_sdk_det(session, det_input)
            infer_t = time.perf_counter() - t0
            
            key = list(output.keys())[0]
            arr = np.array(output[key])
            pm = arr[0, 0] if arr.ndim == 4 else (arr[0] if arr.ndim == 3 else arr)
            rh, rw = resize_info[0], resize_info[1]
            
            print(f"  Input: {det_input.shape} -> Output: {arr.shape} (prob_map: {pm.shape})")
            print(f"  max={arr.max():.6f}, mean={arr.mean():.6f}")
            print(f"  pixels>0.1={(pm>0.1).sum()}, >0.3={(pm>0.3).sum()} / {pm.size}")
            
            # --- 方法A: 直接后处理，调整 resize_info 匹配 prob_map 尺寸 ---
            print(f"\n  [Method A] Adjust resize_info to ({pm.shape[0]},{pm.shape[1]}):")
            adj_ri = (pm.shape[0], pm.shape[1], resize_info[2], orig_h, orig_w)
            boxes_a, scores_a = det_postprocess(
                output, img.shape, adj_ri,
                thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
                unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
                use_dilation=DET_USE_DILATION,
            )
            sa = float(np.mean(scores_a)) if scores_a else 0
            print(f"    => {len(boxes_a)} boxes, avg_score={sa:.4f}")
            all_results[f'{name}_A'] = (len(boxes_a), sa)
            
            # --- 方法B: Upscale prob_map 到 resize 尺寸 ---
            if pm.shape[:2] != (rh, rw):
                print(f"\n  [Method B] Upscale prob_map {pm.shape} -> ({rw},{rh}):")
                pm_up = cv2.resize(pm.astype(np.float32), (rw, rh), interpolation=cv2.INTER_LINEAR)
                up_output = {key: np.array([[pm_up]])}
                boxes_b, scores_b = det_postprocess(
                    up_output, img.shape, resize_info,
                    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
                    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
                    use_dilation=DET_USE_DILATION,
                )
                sb = float(np.mean(scores_b)) if scores_b else 0
                print(f"    => {len(boxes_b)} boxes, avg_score={sb:.4f}")
                all_results[f'{name}_B'] = (len(boxes_b), sb)
                
                # --- 方法C: Upscale 到原图尺寸 ---
                print(f"\n  [Method C] Upscale prob_map -> original ({orig_w},{orig_h}):")
                pm_orig = cv2.resize(pm.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                orig_ri = (orig_h, orig_w, 1.0, orig_h, orig_w)
                orig_output = {key: np.array([[pm_orig]])}
                boxes_c, scores_c = det_postprocess(
                    orig_output, img.shape, orig_ri,
                    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
                    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
                    use_dilation=DET_USE_DILATION,
                )
                sc = float(np.mean(scores_c)) if scores_c else 0
                print(f"    => {len(boxes_c)} boxes, avg_score={sc:.4f}")
                all_results[f'{name}_C'] = (len(boxes_c), sc)
            
            # --- 方法D: 降低阈值测试 ---
            if len(boxes_a) < 50:
                print(f"\n  [Method D] Lower thresholds (thresh=0.1, box_thresh=0.2) on Method A:")
                boxes_d, scores_d = det_postprocess(
                    output, img.shape, adj_ri,
                    thresh=0.10, box_thresh=0.20,
                    unclip_ratio=DET_UNCLIP_RATIO, max_candidates=2000,
                    use_dilation=DET_USE_DILATION,
                )
                sd = float(np.mean(scores_d)) if scores_d else 0
                print(f"    => {len(boxes_d)} boxes, avg_score={sd:.4f}")
                all_results[f'{name}_D_lowthresh'] = (len(boxes_d), sd)
            
            session.Destroy()
            print(f"  Inference time: {infer_t:.3f}s")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
    
    # ===== 4. SDK-CPU + 直接 640×640 输入 (无预处理) =====
    print("\n[4/4] SDK-CPU + Direct 640x640 input...")
    try:
        session = ISession(
            model=det_dlc_path,
            platform="qualcomm",
            framework="snpe",
            runtime="CPU",
            log_level="ERROR",
        )
        ret = session.Initialize()
        if ret == 0:
            tw, th = 640, 640
            resized = cv2.resize(img, (tw, th))
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            normalized = (normalized - mean) / std
            normalized = normalized.transpose((2, 0, 1))
            det_input = np.expand_dims(normalized, axis=0).astype(np.float32)
            ratio = min(th/orig_h, tw/orig_w)
            resize_info = (th, tw, ratio, orig_h, orig_w)
            
            t0 = time.perf_counter()
            output = run_sdk_det(session, det_input)
            infer_t = time.perf_counter() - t0
            
            key = list(output.keys())[0]
            arr = np.array(output[key])
            pm = arr[0, 0] if arr.ndim == 4 else (arr[0] if arr.ndim == 3 else arr)
            print(f"  Output: shape={arr.shape}, max={arr.max():.6f}, mean={arr.mean():.6f}")
            print(f"  ProbMap: {pm.shape}, >0.3={(pm>0.3).sum()}, >0.1={(pm>0.1).sum()}")
            
            boxes, scores = det_postprocess(
                output, img.shape, resize_info,
                thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
                unclip_ratio=DET_UNCLIP_RATIO, max_candidates=DET_MAX_CANDIDATES,
                use_dilation=DET_USE_DILATION,
            )
            s = float(np.mean(scores)) if scores else 0
            print(f"  => {len(boxes)} boxes, avg_score={s:.4f}, time={infer_t:.3f}s")
            all_results['SDK-CPU-640direct'] = (len(boxes), s)
            
            session.Destroy()
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30s} {'Boxes':>6s} {'AvgScore':>9s}")
    print("-" * 50)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)
    for name, (cnt, scr) in sorted_results:
        marker = " <-- TARGET" if 'ORT' in name and 'ref' in name else ""
        print(f"  {name:<28s} {cnt:>6d} {scr:>9.4f}{marker}")

if __name__ == '__main__':
    main()
