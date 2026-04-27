#!/usr/bin/env python3
"""
攻关 Step 1: 针对 DLC Det 的后处理参数调优 + 多尺度检测
目标: 从 56 框提升到接近 88 框
"""
import sys, os, time, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_board import (
    _try_import_sdk, det_preprocess, det_postprocess,
    MODEL_DIR, DET_DLC,
    DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE,
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

def multi_tile_detect(session, img, tile_size=640, overlap=0.3,
                      thresh=0.15, box_thresh=0.25):
    """
    多尺度/多图块检测: 将大图切成重叠的 tile，每个 tile 单独检测，合并结果
    
    原理: DLC 固定输出 640x640，大图直接输入时信息被压缩
         切成 640x640 的 tile 后，每个 tile 获得完整分辨率检测
    """
    h, w = img.shape[:2]
    orig_shape = img.shape
    
    stride = int(tile_size * (1 - overlap))
    all_boxes = []
    all_scores = []
    
    # 计算需要多少个 tile
    nx = max(1, int(np.ceil((w - tile_size) / stride)) + 1) if w > tile_size else 1
    ny = max(1, int(np.ceil((h - tile_size) / stride)) + 1) if h > tile_size else 1
    
    # 调整 stride 使 tile 均匀覆盖图像
    if nx > 1:
        stride_x = (w - tile_size) / (nx - 1)
    else:
        stride_x = w
    if ny > 1:
        stride_y = (h - tile_size) / (ny - 1)
    else:
        stride_y = h
    
    print(f"  Tiling: {nx}x{ny}={nx*ny} tiles, tile={tile_size}, stride=({stride_x:.0f},{stride_y:.0f})")
    
    for row in range(ny):
        for col in range(nx):
            # 计算 tile 坐标 (clip 到边界内)
            x1 = min(int(col * stride_x), w - tile_size) if w > tile_size else 0
            y1 = min(int(row * stride_y), h - tile_size) if h > tile_size else 0
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            
            # 提取 tile
            tile_img = img[y1:y2, x1:x2]
            
            # 预处理 tile (resize 到 tile_size 如果需要)
            th, tw = tile_img.shape[:2]
            if (th, tw) != (tile_size, tile_size):
                tile_resized = cv2.resize(tile_img, (tile_size, tile_size))
            else:
                tile_resized = tile_img
            
            # 归一化
            normalized = tile_resized.astype(np.float32) / 255.0
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            normalized = (normalized - mean) / std
            normalized = normalized.transpose((2, 0, 1))
            det_input = np.expand_dims(normalized, axis=0).astype(np.float32)
            
            # 推理
            output = run_sdk_det(session, det_input)
            
            # 后处理: resize_info 映射回原图坐标
            ratio_h = h / tile_size  # tile 到原图的高度比例
            ratio_w = w / tile_size  # tile 到原图的宽度比例
            resize_info = (tile_size, tile_size, 1.0, tile_size, tile_size)
            
            tile_boxes, tile_scores = det_postprocess(
                output, tile_img.shape, resize_info,
                thresh=thresh, box_thresh=box_thresh,
                unclip_ratio=1.6, max_candidates=500,
                use_dilation=True,
            )
            
            # 将 tile 内坐标映射到原图
            for box, score in zip(tile_boxes, tile_scores):
                global_box = box.copy()
                # 先从 tile 尺寸映射到实际提取区域尺寸
                actual_th, actual_tw = tile_img.shape[:2]
                if (th, tw) != (tile_size, tile_size):
                    # tile 被 resize 过了，先映射回实际 tile 尺寸
                    scale_x = actual_tw / tile_size
                    scale_y = actual_th / tile_size
                    global_box[:, 0] *= scale_x
                    global_box[:, 1] *= scale_y
                
                # 再偏移到原图位置
                global_box[:, 0] += x1
                global_box[:, 1] += y1
                
                # clip 到原图范围
                global_box[:, 0] = np.clip(global_box[:, 0], 0, w)
                global_box[:, 1] = np.clip(global_box[:, 1], 0, h)
                
                all_boxes.append(global_box)
                all_scores.append(score)
    
    return all_boxes, all_scores

def nms_boxes(boxes, scores, iou_threshold=0.5):
    """简单的 NMS 合并重复框"""
    if len(boxes) == 0:
        return [], []
    
    boxes_arr = np.array([b.flatten() for b in boxes])  # (N, 8)
    scores_arr = np.array(scores)
    
    # 计算每个框的面积和置信度排序
    areas = np.zeros(len(boxes))
    for i, b in enumerate(boxes):
        cx = (b[:, 0].min() + b[:, 0].max()) / 2
        cy = (b[:, 1].min() + b[:, 1].max()) / 2
        hw = abs(b[:, 0].max() - b[:, 0].min())
        hh = abs(b[:, 1].max() - b[:, 1].min())
        areas[i] = hw * hh
    
    order = np.argsort(scores_arr)[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        
        # 计算 IoU (简化: 用外接矩形)
        rest = order[1:]
        kept_box = boxes[i]
        
        # 外接矩形 IoU
        x1_i = kept_box[:, 0].min(); y1_i = kept_box[:, 1].min()
        x2_i = kept_box[:, 0].max(); y2_i = kept_box[:, 1].max()
        area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        
        suppress = []
        for j_idx, j in enumerate(rest):
            other_box = boxes[j]
            x1_j = other_box[:, 0].min(); y1_j = other_box[:, 1].min()
            x2_j = other_box[:, 0].max(); y2_j = other_box[:, 1].max()
            area_j = max(0, x2_j - x1_j) * max(0, y2_j - y1_j)
            
            inter_x1 = max(x1_i, x1_j); inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j); inter_y2 = min(y2_i, y2_j)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = area_i + area_j - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > iou_threshold:
                suppress.append(j_idx)
        
        order = np.delete(rest, suppress)
    
    return [boxes[i] for i in keep], [scores[i] for i in keep]

def main():
    SDK = _try_import_sdk()
    if SDK is None:
        print("ERROR: SDK not available")
        return
    
    ISession = SDK['InferenceSession']
    det_dlc_path = os.path.join(MODEL_DIR, DET_DLC)
    
    img = load_image('photos/photo_1.jpg')
    orig_h, orig_w = img.shape[:2]
    print(f"Image: {orig_h}x{orig_w}")
    
    session = ISession(
        model=det_dlc_path, platform="qualcomm",
        framework="snpe", runtime="CPU", log_level="ERROR",
    )
    ret = session.Initialize()
    assert ret == 0, f"Init failed: {ret}"
    
    # ===== 实验1: 不同阈值组合 =====
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Threshold tuning (single-shot DLC)")
    print("=" * 60)
    
    det_input, resize_info = det_preprocess(img, DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, pad_to_square=False)
    output = run_sdk_det(session, det_input)
    
    arr = np.array(output[SDK_DET_OUTPUT])
    pm = arr[0, 0] if arr.ndim == 4 else (arr[0] if arr.ndim == 3 else arr)
    adj_ri = (pm.shape[0], pm.shape[1], resize_info[2], orig_h, orig_w)
    
    best_config = None
    best_count = 0
    
    for thresh in [0.10, 0.15, 0.20]:
        for box_t in [0.20, 0.25, 0.30, 0.35]:
            for unclip in [1.4, 1.6, 1.8, 2.0]:
                boxes, scores = det_postprocess(
                    output, img.shape, adj_ri,
                    thresh=thresh, box_thresh=box_t,
                    unclip_ratio=unclip, max_candidates=2000,
                    use_dilation=True,
                )
                cnt = len(boxes)
                avg_s = float(np.mean(scores)) if scores else 0
                if cnt > best_count:
                    best_count = cnt
                    best_config = (thresh, box_t, unclip, cnt, avg_s)
                if cnt >= 70:  # 接近目标的配置才打印详情
                    print(f"  thresh={thresh:.2f} box_t={box_t:.2f} unclip={unclip:.1f}"
                          f" => {cnt} boxes avg_score={avg_s:.4f}")
    
    t, bt, u, c, s = best_config
    print(f"\n  Best single-shot: thresh={t:.2f}, box_t={bt:.2f}, unclip={u:.1f}"
          f" => {c} boxes, avg_score={s:.4f}")
    
    # ===== 实验2: 多尺度/多图块检测 =====
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-tile detection")
    print("=" * 60)
    
    for tile_size in [480, 640]:
        for overlap in [0.2, 0.3, 0.4]:
            t0 = time.perf_counter()
            raw_boxes, raw_scores = multi_tile_detect(
                session, img, tile_size=tile_size, overlap=overlap,
                thresh=0.15, box_thresh=0.25,
            )
            infer_t = time.perf_counter() - t0
            
            # NMS 合并
            merged_boxes, merged_scores = nms_boxes(raw_boxes, raw_scores, iou_threshold=0.5)
            
            avg_s = float(np.mean(merged_scores)) if merged_scores else 0.0
            print(f"  tile={tile_size} overlap={overlap:.1f}:"
                  f" raw={len(raw_boxes)} -> after_nms={len(merged_boxes)}"
                  f" avg_score={avg_s:.4f}"
                  f" time={infer_t:.2f}s")

if __name__ == '__main__':
    main()
