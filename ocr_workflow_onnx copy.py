"""
=============================================================
  OCR 工作流：图片识别 / 摄像头拍照识别
=============================================================
  依赖：pip install rapidocr opencv-python numpy psutil

  ★★★ 使用方法：只改下面配置区的变量，然后直接运行即可 ★★★
"""

import os
import sys
import time
import cv2
import numpy as np
from rapidocr import RapidOCR


# ============================================================
#  ★★★ 配置区（改这里就行）★★★
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'photos')

# ── 模式选择（填数字）──
MODE = 1   # 1=图片识别    2=摄像头拍照+识别

# ── 模式1：填图片路径（多张放列表里）──
IMAGE_PATHS = [
     "photos/photo_1.jpg",
    # "photos/藿香正气水.jpg",
    #"photos/细菌溶解产物胶囊.jpg",       # ← 示例：只识别这一张
]

# ── 预处理开关（模糊/光照不均时开）──
ENABLE_PREPROCESS = False   # True=启用预处理  False=不用

# ── 过滤关键词（包含这些词的识别结果不打印，如摄像头OSD水印）──
FILTER_KEYWORDS = [
    "MJPG", "fps", "CPU:", "RAM:", "App:", "Photos:", "SPACE:",
    "shot", "quit",
]

# ── OCR 参数（一般不用动）──
OCR_PARAMS = {
    "Det.thresh": 0.20,
    "Det.box_thresh": 0.35,
    "Global.text_score": 0.4,
}

# ── rec 模型选择（量化/原始）──
#   0 = FP32 原始模型 (默认，稳定可靠)
#   1 = INT8 量化模型 (体积略小，精度无损，适合 GPU/NPU 加速设备)
USE_INT8_REC = 0

# ── ONNX Runtime 性能优化 ──
ORT_THREADS = -1          # 推理线程数 (-1=自动/全核)
REC_BATCH_NUM = 6         # rec 批处理大小 (6=默认，文字多可调大; 内存紧张调小)


# ============================================================
#  图像预处理
# ============================================================

def preprocess_image(img):
    """
    ★ OCR 专用图像预处理 ★
    流程：锐化 → CLAHE对比度增强 → 轻微去噪
    
    适用场景：
      - 光照不均 / 有阴影
      - 照片模糊 / 手抖
      - 低光 / 噪点多
    
    注意：清晰照片不需要预处理，PP-OCRv4 内部已有完善处理。
          此函数作为兜底，在识别效果差时启用。
    """
    # Step 1: Unsharp Mask 锐化 (文字边缘更清晰)
    blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
    sharp = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)

    # Step 2: CLAHE 局部对比度增强 (应对光照不均)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 3: 非局部均值去噪 (保留边缘，去除摄像头噪点)
    result = cv2.fastNlMeansDenoisingColored(
        result, None, h=7, templateWindowSize=5, searchWindowSize=15
    )

    return result


def load_image(path):
    """读取图片（支持中文路径）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _write_int8_config(cfg_path, threads=-1):
    """生成 INT8 rec 专用临时配置"""
    int8_model = os.path.join(BASE_DIR, 'pp-ocrv4_rapid_onnx', 'ch_PP-OCRv4_rec_mobile_int8.onnx')
    thr_str = str(threads) if threads > 0 else "-1"
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(f"""Global:
    text_score: 0.4
    use_det: true
    use_cls: true
    use_rec: true
EngineConfig:
    onnxruntime:
        intra_op_num_threads: {thr_str}
        inter_op_num_threads: {thr_str}
        enable_cpu_mem_arena: false
        use_cuda: false
Det:
    engine_type: "onnxruntime"
    lang_type: "ch"
    model_type: "mobile"
    ocr_version: "PP-OCRv4"
    task_type: "det"
Cls:
    engine_type: "onnxruntime"
    lang_type: "ch"
    model_type: "mobile"
    ocr_version: "PP-OCRv4"
    task_type: "cls"
Rec:
    engine_type: "onnxruntime"
    model_path: "{int8_model}"
    rec_img_shape: [3, 48, 320]
    rec_batch_num: {REC_BATCH_NUM}
""")


# ============================================================
#  文本框排序（多列/双栏布局支持）
# ============================================================

def sort_boxes_by_layout(boxes, texts, scores, mode="auto"):
    """
    按阅读顺序重新排序文本框
    
    mode:
      - "auto": 自动检测单列/双列
      - "single": 单列 (从上到下)
      - "double": 双列 (先左列从上到右，再右列从上到下)
    """
    if boxes is None or len(boxes) == 0:
        return boxes, texts, scores

    n = len(boxes)
    centers = np.array([box.mean(axis=0) for box in boxes])
    x_centers = centers[:, 0]
    y_centers = centers[:, 1]

    if mode == "auto":
        x_range = x_centers.max() - x_centers.min()
        y_range = y_centers.max() - y_centers.min()
        if y_range > 0 and (x_range / y_range) > 1.5:
            mode = "double"
        else:
            mode = "single"

    if mode == "single":
        indices = np.argsort(y_centers)

    elif mode == "double":
        x_mid = (x_centers.max() + x_centers.min()) / 2
        left_mask = x_centers <= x_mid
        right_mask = x_centers > x_mid
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]
        left_sorted = left_indices[np.argsort(y_centers[left_indices])]
        right_sorted = right_indices[np.argsort(y_centers[right_indices])]
        indices = np.concatenate([left_sorted, right_sorted])

    else:
        indices = np.arange(n)

    return (
        [boxes[i] for i in indices],
        [texts[i] for i in indices],
        [scores[i] for i in indices]
    )


# ============================================================
#  OCR 核心引擎
# ============================================================

class OCREngine:
    """OCR 引擎封装：模型加载 + 预处理 + 识别 + 排序"""

    def __init__(self, params=None, use_int8_rec=False, ort_threads=-1):
        self.params = params or OCR_PARAMS
        self.use_int8_rec = use_int8_rec
        self.ort_threads = ort_threads
        self._ocr = None
        self._load_time = 0

    def init_model(self):
        """加载 OCR 模型（耗时操作，只需调用一次）"""
        t0 = time.perf_counter()

        rec_label = "INT8(量化)" if self.use_int8_rec else "FP32(原始)"
        print(f"正在加载 PP-OCRv4 模型 [rec: {rec_label}]...")

        if self.use_int8_rec:
            # 使用 INT8 量化模型：通过自定义配置切换
            int8_cfg = os.path.join(BASE_DIR, '_int8_config.yaml')
            _write_int8_config(int8_cfg, self.ort_threads)
            self._ocr = RapidOCR(config_path=int8_cfg, params=self.params)
            if os.path.exists(int8_cfg):
                os.remove(int8_cfg)  # 清理临时文件
        else:
            # FP32 原始模型
            self._ocr = RapidOCR(params=self.params)

        self._load_time = time.perf_counter() - t0
        print(f"模型加载完成 ({self._load_time:.3f}s)\n")

    def recognize(self, image_or_path):
        """
        执行完整 OCR 识别
        
        参数:
          image_or_path: 图片路径(str) 或 numpy 数组(BGR格式)
        
        返回:
          {
            'texts': ['文字1', '文字2', ...],
            'scores': [0.95, 0.88, ...],
            'boxes': [[x1,y1], ...],   # 四角点坐标
            'count': int,
            'avg_score': float,
            'elapsed': float,           # 总耗时
            'elapsed_det': float,
            'elapsed_cls': float,
            'elapsed_rec': float,
          }
          失败返回 None
        """
        if self._ocr is None:
            raise RuntimeError("请先调用 init_model() 初始化模型")

        # 输入可以是路径或 numpy 数组
        if isinstance(image_or_path, str):
            img = load_image(image_or_path)
        else:
            img = image_or_path

        if img is None:
            return None

        # 可选预处理
        if ENABLE_PREPROCESS:
            img = preprocess_image(img)

        # OCR 识别
        output = self._ocr(img)
        if output is None or output.boxes is None or len(output.boxes) == 0:
            return None

        texts = list(output.txts)
        scores = list(output.scores)
        elapse_list = output.elapse_list

        # 根据图片尺寸自动选择排序模式
        h, w = img.shape[:2]
        mode = "double" if w / h > 1.5 else "single"

        # 排序
        boxes, texts, scores = sort_boxes_by_layout(
            output.boxes, texts, scores, mode=mode
        )

        det_t = elapse_list[0] if len(elapse_list) > 0 else 0
        cls_t = elapse_list[1] if len(elapse_list) > 1 else 0
        rec_t = elapse_list[2] if len(elapse_list) > 2 else 0

        return {
            'texts': texts,
            'scores': scores,
            'boxes': boxes,
            'count': len(texts),
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'elapsed': output.elapse,
            'elapsed_det': det_t,
            'elapsed_cls': cls_t,
            'elapsed_rec': rec_t,
        }


# ============================================================
#  摄像头模块
# ============================================================

def find_best_camera():
    """自动扫描所有摄像头，选择最高分辨率的那个"""
    best_idx, best_res = None, 0
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, f = cap.read()
            if ret and f is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                ret2 = cap.read()[0]
                if ret2:
                    mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    mh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    res = mw * mh
                    if res > best_res:
                        best_res, best_idx = res, i
            cap.release()
    return best_idx


def setup_camera(cap):
    """设置摄像头参数 - 优化文字拍摄清晰度"""
    # 分辨率 & 帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # MJPG 格式 (原生支持，清晰度最好)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # 图像优化
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
    cap.set(cv2.CAP_PROP_CONTRAST, 150)
    cap.set(cv2.CAP_PROP_SATURATION, 115)
    try:
        cap.set(cv2.CAP_PROP_SHARPNESS, 200)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_GAIN, 100)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_EXPOSURE, -3)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    except Exception:
        pass


def camera_mode(ocr_engine):
    """摄像头模式：实时预览 + 按空格拍照 + 自动识别"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 50)
    print("  USB Camera - IMX577 (1920x1080 @30fps)")
    print("=" * 50)

    cam_idx = find_best_camera()
    if cam_idx is None:
        print("ERROR: 未找到摄像头!")
        return

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: 无法打开摄像头!")
        return

    setup_camera(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win = 'Camera'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    print(f"[OK] {w}x{h} | SPACE=拍照+识别  Q=退出\n")
    print("提示: 拍照后会自动进行 OCR 识别\n")

    photo_count = 0
    fps_list, t_last, fps_show = [], time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        now = time.time()
        fps_list.append(now)
        fps_list[:] = [t for t in fps_list if now - t < 1.0]
        if now - t_last >= 1.0:
            fps_show = len(fps_list)
            t_last = now

        info = [
            f"{w}x{h}  {fps_show}fps",
            f"Photos: {photo_count}",
            "SPACE=拍照  Q=退出",
        ]
        y = 28
        for line in info:
            cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 120), 2)
            y += 28

        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord(' '):
            # 拍照
            photo_count += 1
            path = os.path.join(SAVE_DIR, f'photo_{photo_count}.jpg')
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if ok:
                with open(path, 'wb') as f:
                    f.write(buf)
                sz = os.path.getsize(path) / 1024
                print(f"\n{'='*60}")
                print(f"  [已保存] photo_{photo_count}.jpg ({sz:.0f}KB)")

                # 立即识别
                result = ocr_engine.recognize(frame)
                if result:
                    print_result(result, path)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#  结果输出
# ============================================================

def print_result(result, source_name=""):
    """格式化打印 OCR 识别结果"""
    # 过滤掉包含关键词的文字（如摄像头OSD水印）
    filtered = []
    for text, score in zip(result['texts'], result['scores']):
        if any(kw in text for kw in FILTER_KEYWORDS):
            continue
        filtered.append((text, score))

    count = len(filtered)
    total = result['count']
    skipped = total - count

    print(f"  ┌─ 识别到 {total} 段文字（过滤 {skipped} 条水印），显示 {count} 段，平均置信度 {result['avg_score']:.3f}")
    if count == 0:
        print("  │  （全部被过滤或无文字）")
    else:
        print("  │")
        for text, score in filtered:
            tag = "" if score >= 0.95 else f"  [{score:.2f}]"
            print(f"  │  {text}{tag}")
    print("  │")
    print(f"  └─ 耗时: 检测{result['elapsed_det']:.2f}s + 分类{result['elapsed_cls']:.2f}s + 识别{result['elapsed_rec']:.2f}s = 总计{result['elapsed']:.2f}s")
    print()


# ============================================================
#  主入口
# ============================================================

def batch_mode(ocr_engine, image_paths):
    """批量识别模式"""
    total_start = time.perf_counter()
    total_images = 0

    for path in image_paths:
        if not os.path.exists(path):
            print(f"[跳过] 文件不存在: {path}")
            continue

        print(f"{'='*60}")
        print(f"  图片: {path}")
        print(f"{'='*60}")

        result = ocr_engine.recognize(path)
        if result:
            print_result(result, path)
            total_images += 1
        else:
            print("  [未识别到文字]\n")

    total_time = time.perf_counter() - total_start
    print(f"{'='*60}")
    print(f"[完成] 共识别 {total_images} 张图片, 总耗时 {total_time:.3f}s")
    if total_images > 0:
        print(f"        平均每张 {total_time/total_images:.3f}s")
    print(f"{'='*60}")


def main():
    # 初始化 OCR 引擎
    engine = OCREngine(
        params=OCR_PARAMS,
        use_int8_rec=(USE_INT8_REC == 1),
        ort_threads=ORT_THREADS,
    )
    engine.init_model()

    # ── 根据配置区 MODE 变量选择模式 ──
    if MODE == 2:
        # ════════════════════ 模式2: 摄像头 ════════════════════
        print("\n[模式2] 摄像头拍照识别")
        camera_mode(engine)

    elif MODE == 1:
        # ════════════════════ 模式1: 图片识别 ════════════════════
        if not IMAGE_PATHS or all(p.strip().startswith('#') for p in IMAGE_PATHS):
            print("!!! 配置区 IMAGE_PATHS 为空，请填入图片路径 !!!")
            return
        # 过滤掉注释行（以#开头）
        paths = [p for p in IMAGE_PATHS if p and not p.strip().startswith('#')]
        print(f"\n[模式1] 图片识别 ({len(paths)} 张)")
        batch_mode(engine, paths)

    else:
        print(f"!!! MODE 只能填 1 或 2，你填了 {MODE} !!!")


if __name__ == '__main__':
    main()
