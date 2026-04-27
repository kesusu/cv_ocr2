#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EasyOCR ONNX w8a8 模型测试脚本
对比 EasyOCR (w8a8 INT8) vs PaddleOCR v4 (FP32) 的识别效果
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort


# ============================================================
#  配置
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "easyocr-onnx-float", "easyocr-onnx-float")

# EasyOCR 英文字符表 (97类: 0=blank, 1-96=可打印ASCII)
EASYOCR_CHARS = (
    "\x00-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n'
)

# PaddleOCR 字典路径
PPocr_KEYS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pp-ocrv4_rapid_onnx", "ppocr_keys_v1.txt")


# ============================================================
#  EasyOCR w8a8 推理器
# ============================================================
class EasyOCRW8A8Inference:
    """基于 easyocr-onnx-w8a8 模型的 OCR 推理"""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = MODEL_DIR

        det_path = os.path.join(model_dir, "detector.onnx")
        rec_path = os.path.join(model_dir, "recognizer.onnx")
        assert os.path.exists(det_path), f"[ERROR] 找不到检测模型: {det_path}"
        assert os.path.exists(rec_path), f"[ERROR] 找不到识别模型: {rec_path}"

        print(f"[INFO] 加载检测模型: {det_path}")
        self.det_session = ort.InferenceSession(det_path,
                                                providers=['CPUExecutionProvider'])
        print(f"[INFO] 加载识别模型: {rec_path}")
        self.rec_session = ort.InferenceSession(rec_path,
                                                providers=['CPUExecutionProvider'])

        # Detector 输入输出信息
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_input_shape = self.det_session.get_inputs()[0].shape  # [1,3,608,800]
        self.det_output_names = [o.name for o in self.det_session.get_outputs()]

        # Recognizer 输入输出信息
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_input_shape = self.rec_session.get_inputs()[0].shape  # [1,1,64,800]
        self.rec_output_name = self.rec_session.get_outputs()[0].name
        # Float 模型，无需反量化
        print("[INFO] Recognizer: float 模型")

        self.blank_idx = 0  # CTC blank
        self.chars = EASYOCR_CHARS

    # ---- 预处理 ----

    def resize_pad_image(self, img, target_size, pad_value=0,
                         return_scale=True):
        """
        将图片 resize + pad 到目标尺寸，保持长宽比
        Returns:
            padded_img, scale, (pad_w, pad_h)
        """
        th, tw = target_size[:2]
        h, w = img.shape[:2]

        # 计算缩放比例（保持比例）
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Pad
        pad_top = (th - nh) // 2
        pad_bottom = th - nh - pad_top
        pad_left = (tw - nw) // 2
        pad_right = tw - nw - pad_left

        if len(resized.shape) == 2:
            resized = resized[:, :, np.newaxis]

        n_channels = resized.shape[2] if len(resized.shape) == 3 else 1
        canvas = np.full((th, tw, n_channels), pad_value, dtype=resized.dtype)
        canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

        if return_scale:
            return canvas, scale, (pad_left, pad_top)
        return canvas

    def detector_preprocess(self, img_rgb):
        """
        预处理用于 CRAFT 文本检测器
        - 输入: RGB 图像 (H,W,3), uint8, [0,255]
        - 输出: float tensor (1,3,608,800), 归一化到 [0,1]
        """
        det_input, scale, (pad_w, pad_h) = self.resize_pad_image(
            img_rgb, (608, 800), pad_value=0
        )
        det_input = det_input.astype(np.float32) / 255.0  # 归一化
        det_input = np.transpose(det_input, (2, 0, 1))   # HWC -> CHW
        det_input = np.expand_dims(det_input, axis=0)     # 加 batch 维度
        return det_input, scale, (pad_w, pad_h)

    def recognizer_preprocess(self, img_gray):
        """
        预处理用于 CRNN 文本识别器
        - 输入: 灰度图 (H,W), uint8, [0,255]
        - 输出: float tensor (1,1,64,800), 标准化
        """
        rec_input = self.resize_pad_image(
            img_gray, (64, 800), pad_value=0,
            return_scale=False
        ).astype(np.float32)

        # 标准化: (x - 128) / 128  => 映射到 [-1, 1]
        rec_input = (rec_input - 128.0) / 128.0

        # 转为 (1,1,64,800)
        if len(rec_input.shape) == 2:
            rec_input = rec_input[np.newaxis, :, :]
        else:
            rec_input = np.transpose(rec_input, (2, 0, 1))
        rec_input = np.expand_dims(rec_input, axis=0)

        return rec_input

    # ---- 后处理 / CRAFT ----

    def detector_postprocess(self, det_output, score_threshold=0.5,
                             link_threshold=0.4, min_box_size=10):
        """
        从 CRAFT 输出中提取文本框
        det_output: dict 或 list, 包含 score_map 和 link_map
                     每个 shape 为 (B, H, W) 或 (B, H, W, 1)
        Returns:
            boxes: list of (x1,y1,x2,y2) 或 (4,2) 多边形坐标
            scores: list of float
        """
        # 提取 score_map 和 link_map
        out_tensor = det_output[0]  # list, 取第一个输出
        print("[DEBUG] det_output raw shape:", out_tensor.shape)

        # 处理不同的输出格式
        if out_tensor.shape[-1] == 2 or (len(out_tensor.shape) == 4 and out_tensor.shape[1] == 2):
            # 格式: (B, 2, H, W) 或 (B, H, W, 2)
            if out_tensor.shape[1] == 2:
                score_map = out_tensor[0, 0]  # (H, W)
                link_map = out_tensor[0, 1]    # (H, W)
            else:
                score_map = out_tensor[0, :, :, 0]
                link_map = out_tensor[0, :, :, 1]
        else:
            print("[WARN] 无法解析检测输出格式，使用默认分割")
            mid = out_tensor.shape[1] // 2
            score_map = out_tensor[0, :mid, :]  # 前一半作为 score
            link_map = out_tensor[0, mid:, :]   # 后一半作为 link

        print("[DEBUG] score_map shape:", score_map.shape,
              "range: [{:.4f}, {:.4f}]".format(score_map.min(), score_map.max()))
        print("[DEBUG] link_map shape:", link_map.shape,
              "range: [{:.4f}, {:.4f}]".format(link_map.min(), link_map.max()))

        # 简化版后处理: 使用阈值 + 连通域提取文本区域
        text_mask = (score_map > score_threshold).astype(np.uint8) * 255
        link_mask = (link_map > link_threshold).astype(np.uint8) * 255

        # 组合 mask: 同时满足 score 和 link 条件
        combined = cv2.bitwise_and(text_mask, link_mask)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        scores = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_box_size:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x+w, y+h])
            # 取该区域的平均 score 作为置信度
            mask_region = score_map[y:y+h, x:x+w]
            scores.append(float(np.mean(mask_region)))

        return boxes, scores

    # ---- 后处理 / CRNN 解码 ----

    def ctc_decode_greedy(self, probs, blank_idx=0):
        """
        CTC greedy 解码
        probs: (T, C) 概率矩阵
        Returns: text string
        """
        best_path = np.argmax(probs, axis=1)
        chars = []
        prev = -1
        for idx in best_path:
            if idx != blank_idx and idx != prev:
                if idx < len(self.chars):
                    c = self.chars[idx]
                    if c.strip() or c not in ['\x00', '\t', '\n']:
                        chars.append(c)
            prev = idx
        return ''.join(chars)

    def crop_text_regions(self, img, boxes, padding=0.3):
        """
        根据检测框裁剪文本区域
        Returns:
            crops: list of cropped grayscale images
        """
        crops = []
        h_orig, w_orig = img.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]

            # 添加 padding
            pw = int((x2 - x1) * padding)
            ph = int((y2 - y1) * padding)
            x1 = max(0, x1 - pw)
            y1 = max(0, y1 - ph)
            x2 = min(w_orig, x2 + pw)
            y2 = min(h_orig, y2 + ph)

            # 裁剪并转为灰度图
            if len(img.shape) == 3:
                crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            else:
                crop = img[y1:y2, x1:x2]

            crops.append(crop)
        return crops

    # ---- 完整推理流程 ----

    def infer(self, image_path, vis_path=None):
        """
        完整的 OCR 推理流程
        Args:
            image_path: 输入图像路径
            vis_path: 可视化结果保存路径 (可选)

        Returns:
            dict: {
                'texts': list[str],
                'boxes': list[list],
                'scores': list[float],
                'elapsed': float
            }
        """
        t_start = time.time()

        # 1. 读取图像
        if isinstance(image_path, str):
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"无法读取图像: {image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_path.copy()
            if len(img_rgb.shape) == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

        orig_h, orig_w = img_rgb.shape[:2]
        print("[INFO] 输入图像尺寸: {}x{}".format(orig_w, orig_h))

        # 2. 检测器推理
        t_det = time.time()
        det_input, scale, (pad_w, pad_h) = self.detector_preprocess(img_rgb)
        det_output = self.det_session.run(
            self.det_output_names,
            {self.det_input_name: det_input}
        )
        t_det_end = time.time()
        print("[DEBUG] 检测耗时: {:.3f}s".format(t_det_end - t_det))

        # 3. 检测后处理 → 得到文本框
        boxes, scores = self.detector_postprocess(
            det_output, score_threshold=0.3, link_threshold=0.25
        )

        if len(boxes) == 0:
            print("[WARN] 未检测到文本区域")
            return {'texts': [], 'boxes': [], 'scores': [],
                    'elapsed': time.time() - t_start}

        print("[INFO] 检测到 %d 个文本区域" % len(boxes))

        # 将框坐标从检测输入尺寸映射回原图
        inv_scale = 1.0 / scale
        mapped_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            mx1 = max(0, (x1 - pad_w) * inv_scale)
            my1 = max(0, (y1 - pad_h) * inv_scale)
            mx2 = min(orig_w, (x2 - pad_w) * inv_scale)
            my2 = min(orig_h, (y2 - pad_h) * inv_scale)
            mapped_boxes.append([mx1, my1, mx2, my2])

        # 4. 裁剪文本区域
        crops = self.crop_text_regions(img_rgb, mapped_boxes, padding=0.2)

        # 5. 逐个识别
        texts = []
        final_scores = []
        for i, crop in enumerate(crops):
            rec_input = self.recognizer_preprocess(crop)
            rec_output = self.rec_session.run(
                [self.rec_output_name],
                {self.rec_input_name: rec_input}
            )[0]  # shape: (1, T, 97) float

            probs = rec_output[0]  # (T, 97)
            text = self.ctc_decode_greedy(probs, blank_idx=self.blank_idx)
            confidence = float(np.max(probs, axis=1).mean())

            if text.strip():
                texts.append(text)
                final_scores.append(confidence)
                print("  [%d] \"%s\" (%.3f)" % (i+1, text, confidence))

        elapsed = time.time() - t_start
        print("[INFO] 总耗时: %.3fs" % elapsed)

        # 6. 可视化
        if vis_path is not None:
            vis_img = img_bgr.copy()
            for j, (box, txt, sc) in enumerate(zip(mapped_boxes, texts, final_scores)):
                x1, y1, x2, y2 = [int(v) for v in box]
                color = (0, 255, 0)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                label = "%s %.2f" % (txt, sc)
                cv2.putText(vis_img, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(vis_path, vis_img)
            print("[INFO] 可视化已保存: %s" % vis_path)

        return {
            'texts': texts,
            'boxes': mapped_boxes,
            'scores': final_scores,
            'elapsed': elapsed
        }


# ============================================================
#  PaddleOCR v4 推理器 (FP32 ONNX, 用于对比)
# ============================================================
class PaddleOCRInference:
    """PaddleOCR v4 rapid 版本 (ONNX FP32)"""

    def __init__(self):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pp-ocrv4_rapid_onnx")
        det_model = os.path.join(base, "ch_PP-OCRv4_det_mobile.onnx")
        rec_model = os.path.join(base, "ch_PP-OCRv4_rec_mobile.onnx")
        dict_path = PPocr_KEYS

        assert os.path.exists(det_model), f"找不到: {det_model}"
        assert os.path.exists(rec_model), f"找不到: {rec_model}"

        print("[INFO] PaddleOCR 检测: %s" % det_model)
        self.det_sess = ort.InferenceSession(
            det_model, providers=['CPUExecutionProvider'])
        print("[INFO] PaddleOCR 识别: %s" % rec_model)
        self.rec_sess = ort.InferenceSession(
            rec_model, providers=['CPUExecutionProvider'])

        self.det_input_name = self.det_sess.get_inputs()[0].name
        self.rec_input_name = self.rec_sess.get_inputs()[0].name
        self.rec_output_name = self.rec_sess.get_outputs()[0].name

        # 加载字典
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_list = [line.strip('\n') for line in f]
        print("[INFO] PaddleOCR 字典大小: %d" % len(self.char_list))

    def preprocess_ppocr_det(self, img, limit_side_len=960):
        """PP-OCRv4 检测预处理"""
        h, w = img.shape[:2]
        ratio = 1.0
        if max(h, w) > limit_side_len:
            ratio = limit_side_len / max(h, w)
        resize_w, resize_h = int(w * ratio), int(h * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        resized = cv2.resize(img, (resize_w, resize_h))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_float = (resized.astype(np.float32) / 255.0 - mean) / std
        img_float = img_float.transpose(2, 0, 1)[np.newaxis, :]
        return img_float.astype(np.float32), (ratio, ratio)

    def postprocess_ppocr_det(self, output, ratio=(1, 1),
                              det_db_thresh=0.3, det_db_box_thresh=0.5):
        """PP-OCRv4 检测 DB 后处理"""
        pred = output[0][0]
        bitmap = (pred > det_db_thresh).astype(np.uint8) * 255
        boxes_list = []
        scores = []

        if bitmap.max() == 0:
            return [], []

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box[:, 0] /= ratio[1]
            box[:, 1] /= ratio[0]
            box_list = box.flatten().tolist()
            boxes_list.append(box_list)
            scores.append(float(pred[
                int(min(box[:, 1])):int(max(box[:, 1])),
                int(min(box[:, 0])):int(max(box[:, 0]))
            ].mean()))

        indices = sorted(range(len(scores)), key=lambda i: boxes_list[i][1])
        boxes_list = [boxes_list[i] for i in indices]
        scores = [scores[i] for i in indices]
        return boxes_list, scores

    def preprocess_ppocr_rec(self, img_crop, wh_ratio=3.23):
        """PP-OCRv4 识别预处理"""
        h, w = img_crop.shape[:2]
        rw = max(100, int(h * wh_ratio))
        resized = cv2.resize(img_crop, (rw, h))
        padded = np.zeros((h, max(rw, h * 5), 3), dtype=np.uint8)
        padded[:, :rw] = resized
        padded = padded.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32) / 255.
        return padded

    def ctc_decode_ppocr(self, preds, idx2char):
        """CTC greedy decode for PaddleOCR"""
        text = ''
        last = 0
        for idx in preds:
            if idx > 0 and idx != last:
                if idx < len(idx2char):
                    text += idx2char[idx]
            last = idx
        return text

    def infer(self, image_path):
        """完整 PaddleOCR 推理"""
        t = time.time()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取: " + image_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 检测
        det_input, ratio = self.preprocess_ppocr_det(rgb)
        det_out = self.det_sess.run(None, {self.det_input_name: det_input})
        boxes, scores = self.postprocess_ppocr_det(det_out, ratio)

        if not boxes:
            return {'texts': [], 'boxes': [], 'scores': [],
                    'elapsed': time.time()-t}

        # 排序 + 识别
        indices = sorted(range(len(boxes)), key=lambda b: boxes[b][1])
        texts = []
        for idx in indices:
            box = np.array(boxes[idx]).reshape(-1, 2).astype(np.int32)
            xmin, ymin = box.min(axis=0)
            xmax, ymax = box.max(axis=0)
            crop = img[max(ymin,0):ymax+1, max(xmin,0):xmax+1]
            if crop.size == 0:
                continue
            rec_in = self.preprocess_ppocr_rec(crop)
            rec_out = self.rec_sess.run(None, {self.rec_input_name: rec_in})[0][0]
            text = self.ctc_decode_ppocr(rec_out.argmax(axis=1), self.char_list)
            if text.strip():
                texts.append(text)

        return {'texts': texts, 'boxes': boxes, 'scores': scores,
                'elapsed': time.time()-t}


# ============================================================
#  主函数：对比测试
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='EasyOCR vs PaddleOCR 对比测试')
    parser.add_argument('image', help='测试图片路径')
    parser.add_argument('--model-dir', default=None, help='EasyOCR ONNX 模型目录')
    parser.add_argument('--vis-easy', default=None, help='EasyOCR 结果保存路径')
    parser.add_argument('--vis-paddle', default=None, help='PaddleOCR 结果保存路径')
    parser.add_argument('--no-paddle', action='store_true', help='跳过 PaddleOCR')
    args = parser.parse_args()

    print("=" * 60)
    print("EasyOCR w8a8 (ONNX) vs PaddleOCR v4 (FP32) 对比测试")
    print("=" * 60)
    print("\n>>> 测试图片: %s" % args.image)

    # --- EasyOCR w8a8 ---
    print("-"*60)
    print("[1/2] EasyOCR w8a8 (ONNX) 推理...")
    print("-"*60)
    easy_inf = EasyOCRW8A8Inference(args.model_dir)
    r_easy = easy_inf.infer(args.image, vis_path=args.vis_easy)

    # --- PaddleOCR ---
    r_paddle = None
    if not args.no_paddle:
        print("-"*60)
        print("[2/2] PaddleOCR v4 (FP32) 推理...")
        print("-"*60)
        try:
            paddle_inf = PaddleOCRInference()
            r_paddle = paddle_inf.infer(args.image)
        except Exception as e:
            print("[WARN] PaddleOCR 运行失败: %s" % str(e))

    # --- 对比总结 ---
    print("\n" + "=" * 60)
    print("对比总结:")
    print("=" * 60)
    print("%-22s %6s %10s" % ("模型", "文本数", "耗时"))
    print("-" * 44)
    print("%-22s %6d %9.3fs" % ("EasyOCR w8a8 (ONNX)", len(r_easy['texts']), r_easy['elapsed']))
    if r_paddle:
        print("%-22s %6d %9.3fs" % ("PaddleOCR v4 (FP32)", len(r_paddle['texts']), r_paddle['elapsed']))

    print("\nEasyOCR 文字内容:")
    for i, t in enumerate(r_easy['texts']):
        print("  [%d] %s" % (i+1, t))
    if r_paddle:
        print("\nPaddleOCR 文字内容:")
        for i, t in enumerate(r_paddle['texts']):
            print("  [%d] %s" % (i+1, t))


if __name__ == '__main__':
    main()
