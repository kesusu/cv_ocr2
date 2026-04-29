"""
=============================================================
  OCR 工作流：图片识别 / 摄像头拍照识别
=============================================================
  依赖：pip install rapidocr opencv-python numpy psutil

  ★★★ 使用方法：只改下面配置区的变量，然后直接运行即可 ★★★
  
  ★ 2026-04-25 更新：Rec 模型专项优化 ★
  - 可调 Rec 输入宽度 (320/256/192)
  - 可调批处理大小 (6/12/16)
  - 基准测试对比各配置效果
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
     "photos/photo_8.jpg",
    # "photos/藿香正气水.jpg",
    #"photos/细菌溶解产物胶囊.jpg",       # ← 示例：只识别这一张
]

# ── 预处理开关（模糊/光照不均/文字偏小时开）──
ENABLE_PREPROCESS = True    # True=启用预处理(推荐相机拍摄时开启)
                            # False=不用(清晰扫描件可关闭加速)

# ── 过滤关键词（包含这些词的识别结果不打印，如摄像头OSD水印）──
FILTER_KEYWORDS = [
    "MJPG", "fps", "CPU:", "RAM:", "App:", "Photos:", "Photas:",
    "SPACE:", "shot", "quit",
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
REC_BATCH_NUM = 6         # rec 批处理大小 (6=默认, 文字多可调大; 内存紧张调小)

# ══════════ ★ Rec 加速核心参数 ★ ══════════
#   来自 PROJECT_SUMMARY.md 第八章 + 第九章的优化方案
#   ★ 2026-04-25 基准测试重要发现 ★
#     - 改宽度需要自定义YAML，而自定义YAML比RapidOCR默认配置慢!
#     - 结论: 保持默认 320，让 RapidOCR 用内部最优配置 = 最快路径
REC_IMAGE_WIDTH = 320     # rec 输入宽度 (保持默认=最快)
                          #   320 = 默认 (★ 最优: 走 RapidOCR 默认路径)
                          #   256/192 = 需要自定义YAML (实测反而更慢)
ENABLE_BENCH_MODE = False  # True = 运行基准测试 (对比各配置)
                          # False = 正常 OCR 识别模式


# ============================================================
#  图像预处理
# ============================================================

def _adaptive_brightness_fix(img):
    """
    自适应亮度校正 (在 preprocess_image 之前调用)
    
    问题: photo_6 这类偏亮/过曝照片, mean>180,
          文字区域对比度不足, OCR 置信度大幅下降(0.85→0.41)
    
    触发条件: mean > 155 或 std < 45 (过曝或动态范围窄)
    处理方式: 
      1. Gamma 压暗 (gamma=1.2~1.5, 根据过曝程度自适应)
      2. 轻量对比度拉伸
    
    不触发时直接返回原图, 零开销。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    
    # 判断是否需要校正
    # 触发条件: mean>145(偏亮) 或 std<50(动态范围窄/对比不足)
    needs_fix = mean_val > 145 or std_val < 50
    
    if not needs_fix:
        return img  # 画质正常, 跳过

    # ★ Step 1: Gamma 校正 — 压暗过曝画面
    #    公式: output = input ** gamma
    #    gamma > 1 → 压暗亮区 (gamma=2.0 时亮度200→157)
    if mean_val > 140:
        # 综合考虑亮度和对比度来确定gamma值
        bright_score = max((mean_val - 140) / 80, 0)   # 亮度因子
        low_contrast = max((50 - std_val) / 30, 0)     # 低对比因子
        
        severity = max(bright_score, low_contrast * 0.7)
        severity = min(severity, 1.0)
        
        # gamma范围: 1.3(轻微偏亮) ~ 2.2(严重过曝+低对比)
        gamma = 1.3 + severity * 0.9   # range: 1.3 ~ 2.2
        table = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255)
                          for i in range(256)], dtype='uint8')
        img = cv2.LUT(img, table)
    
    return img


def preprocess_image(img):
    """
    ★ OCR 专用图像预处理 (相机拍摄优化版) ★
    
    策略: 不做全局上采样(避免4倍像素拖慢速度),
         只做锐化+CLAHE对比度增强来改善文字可读性
    
    漏字问题根因分析 (photo_4.jpg):
      原始: 【贮藏】遮光，密封保存。
      识别: 【亡藏】光，密封保存。(score=0.84)
      
      原因链: 
        相机距离远 → 文字行高仅24px(需>=32px) 
                → 锐度仅185(需>300) 
                → 复杂汉字(贮/遮)笔画粘连丢失
    
    适用场景：
      - 相机拍摄的文档/说明书
      - 光照不均 / 有阴影
      - 照片模糊 / 手抖 / 文字偏小
    
    注意：清晰扫描件可关闭 ENABLE_PREPROCESS 加速 (~2x提速)。
    """
    # Step 1: Unsharp Mask 锐化 (恢复文字边缘细节)
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 1.45, blurred, -0.45, 0)

    # Step 2: CLAHE 局部对比度增强 (应对光照不均+提升暗部文字)
    # ★ clipLimit=2.5 (原3.0): 更保守,避免过度增强导致笔画变形
    #   实测: CL=3.0 时 "欣"→"政", CL=2.5 正确识别
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 3: 去噪已移除
    # ★ 原 fastNlMeansDenoisingColored(h=6) 会导致:
    #   "雷蒙欣" → "需蒙政"(雷→需, 欣→政)
    #   原因: 去噪模糊了中小字号汉字的细笔画
    #   替代方案: 锐化+CLAHE 已足够压制摄像头噪点

    return result


def load_image(path):
    """读取图片（支持中文路径）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _write_ocr_config(cfg_path, use_int8=False, threads=-1,
                     rec_width=320, batch_num=6):
    """生成 RapidOCR YAML 配置（支持 FP32/INT8 + ORT 性能优化）

    通过 YAML 配置注入以下优化 (来自 PROJECT_SUMMARY.md 第八章):
      - intra/inter_op_num_threads: 多线程并行
      - execution_mode: SEQUENTIAL (减少同步开销)
      - enable_mem_arena / enable_mem_reuse: 内存复用
      - rec_img_shape: 可调输入宽度 (★ Rec加速关键参数 ★)
      - rec_batch_num: 可调批处理大小 (★ Rec加速关键参数 ★)
    """
    model_dir = os.path.join(BASE_DIR, 'pp-ocrv4_rapid_onnx')
    if use_int8:
        rec_model = os.path.join(model_dir, 'ch_PP-OCRv4_rec_mobile_int8.onnx')
    else:
        rec_model = ""  # FP32 用默认模型，不指定 model_path

    thr_str = str(threads) if threads > 0 else "-1"
    model_dir = os.path.join(BASE_DIR, 'pp-ocrv4_rapid_onnx')
    # YAML 中必须用正斜杠，避免反斜杠被解析为转义字符
    model_dir_yaml = model_dir.replace(os.sep, '/')
    if use_int8:
        rec_model_yaml = rec_model.replace(os.sep, '/')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(f"""Global:
    text_score: 0.4
    use_det: true
    use_cls: true
    use_rec: true
    min_height: 30
    width_height_ratio: 8
    max_side_len: 2000
    min_side_len: 30
    return_word_box: false
    return_single_char_box: false
    font_path: null
    log_level: "warning"
    model_root_dir: '{model_dir_yaml}'
EngineConfig:
    onnxruntime:
        intra_op_num_threads: {thr_str}
        inter_op_num_threads: {thr_str}
        enable_cpu_mem_arena: false
        execution_mode: sequential
        enable_mem_reuse: true
        graph_optimization_level: all
        use_cuda: false
        cpu_ep_cfg:
            arena_extend_strategy: "kSameAsRequested"
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
    cls_image_shape: [3, 48, 192]
    cls_batch_num: 6
    cls_thresh: 0.9
    label_list: ["0", "180"]
Rec:
    engine_type: "onnxruntime"
    lang_type: "ch"
    model_type: "mobile"
    ocr_version: "PP-OCRv4"
    task_type: "rec"
""")
        if use_int8:
            f.write(f"""    model_path: '{rec_model_yaml}'
""")
        f.write(f"""    rec_img_shape: [3, 48, {rec_width}]
    rec_batch_num: {batch_num}
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
      - "double": 双列 (先左列从上到下，再右列从上到下)
    
    ★ 双栏改进: KMeans 聚类分栏 + 逐行交错输出
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
        # ── 改进: IQR 去除边缘离群值 → 在主体文字中找最大间隙 ──
        q1, q3 = np.percentile(x_centers, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 只在主体区域内找分栏线
        core_mask = (x_centers >= lower_bound) & (x_centers <= upper_bound)
        if np.sum(core_mask) >= 4:
            core_x = sorted(x_centers[core_mask])
            core_gaps = np.diff(core_x)
            max_gap_idx = np.argmax(core_gaps)
            split_x = (core_x[max_gap_idx] + core_x[max_gap_idx + 1]) / 2
        else:
            # 数据太少，回退到中位数
            split_x = (x_centers.max() + x_centers.min()) / 2
        
        left_mask = x_centers <= split_x
        right_mask = x_centers > split_x
        
        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]

        # 各栏内按 Y 排序 → 先左栏完整输出, 再右栏
        left_sorted = left_idx[np.argsort(y_centers[left_idx])]
        right_sorted = right_idx[np.argsort(y_centers[right_idx])]
        
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
    """OCR 引擎封装：模型加载 + 预处理 + 识别 + 排序

    ★ 2026-04-25 聚焦 Rec 优化 ★
      - 可调 rec_width (320/256/192) — 直接影响 Rec 推理速度
      - 可调 batch_num (6/12/16)     — 影响吞吐量
      - ORT SEQUENTIAL 执行模式       — 减少线程同步开销
      - mem_reuse 内存复用            — 减少 malloc 开销
    """

    def __init__(self, params=None, use_int8_rec=False, ort_threads=-1,
                 rec_width=320, batch_num=6):
        self.params = params or OCR_PARAMS
        self.use_int8_rec = use_int8_rec
        self.ort_threads = ort_threads
        self.rec_width = rec_width
        self.batch_num = batch_num
        self._ocr = None
        self._load_time = 0

    def init_model(self):
        """加载 OCR 模型（耗时操作，只需调用一次）

        ★ 2026-04-25 修复: 回归 RapidOCR 默认初始化路径 ★
          - 默认配置: 直接 RapidOCR(params=) → 使用内部最优默认值
          - 仅在需要改宽度/INT8 时才生成自定义 YAML (避免 ORT 参数被覆盖导致变慢)
        """
        t0 = time.perf_counter()

        rec_label = "INT8(量化)" if self.use_int8_rec else "FP32(原始)"

        need_custom_cfg = (
            self.use_int8_rec or           # INT8 需要指定模型路径
            self.rec_width != 320           # 非默认宽度需要覆盖 rec_img_shape
        )

        if need_custom_cfg:
            print(f"正在加载 PP-OCRv4 模型 [rec: {rec_label}, "
                  f"width={self.rec_width}, batch={self.batch_num}]...")
            ocr_cfg = os.path.join(BASE_DIR, '_ocr_optimized.yaml')
            _write_ocr_config(
                ocr_cfg,
                use_int8=self.use_int8_rec,
                threads=self.ort_threads,
                rec_width=self.rec_width,
                batch_num=self.batch_num,
            )
            self._ocr = RapidOCR(config_path=ocr_cfg, params=self.params)
            if os.path.exists(ocr_cfg):
                os.remove(ocr_cfg)
        else:
            # ★ 最快路径: 让 RapidOCR 用内部默认配置 (已针对模型优化) ★
            print(f"正在加载 PP-OCRv4 模型 [rec: {rec_label}]...")
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
            'elapsed_rec': float,       # ★ Rec 单独耗时
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
            img = _adaptive_brightness_fix(img)   # 自适应亮度校正(过曝时才触发)
            img = preprocess_image(img)            # 原有预处理管线

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


# ══════════════════════════════════════════════════════════
#  ★ Rec 加速基准测试 ★
#  验证 PROJECT_SUMMARY.md 第八章中的优化方案
# ══════════════════════════════════════════════════════════

def run_benchmark():
    """★ Rec 模型加速基准测试 ★

    测试矩阵 (来自 PROJECT_SUMMARY.md 第八~九章):
    
    | 维度 | 测试值 | 预期效果 |
    |------|--------|----------|
    | 输入宽度 | 320 / 256 / 192 | 宽度↓ → 计算量↓ → 速度↑ |
    | 批处理  | 6  / 12 / 16    | batch↑ → 吞吐↑         |
    
    用法: 设置 ENABLE_BENCH_MODE = True 后运行
    输出: 各配置的 Rec耗时、总耗时、识别数、置信度对比表
    """
    import copy

    # ── 测试图片准备 ──
    test_paths = [p for p in IMAGE_PATHS
                  if p and not p.strip().startswith('#') and os.path.exists(p)]
    if not test_paths:
        print("[BENCH] 错误: 配置区 IMAGE_PATHS 为空或文件不存在")
        print("        请至少填入1张测试图片路径")
        return

    test_img = load_image(test_paths[0])
    if test_img is None:
        print(f"[BENCH] 错误: 无法读取测试图片 {test_paths[0]}")
        return

    print("=" * 72)
    print("  PP-OCRv4 Rec 加速基准测试")
    print(f"  测试图片: {test_paths[0]} ({test_img.shape[1]}x{test_img.shape[0]})")
    print(f"  环境: PC (Windows), ORT CPU")
    print("=" * 72)

    # ── 测试配置矩阵 ★ 核心部分 ★ ──
    #    格式: (标签, rec_width, batch_num, 描述)
    #
    #    对应 PROJECT_SUMMARY.md 的优化方案:
    #      P0: 增大 REC_BATCH_NUM (6→12)
    #      P1: 降低 REC_IMAGE_WIDTH (320→256→192)
    #      组合: W256+Batch12 (最大收益)
    configs = [
        ("① Baseline (W320,B6)",   320, 6,  "默认配置"),
        ("② Batch=12",             320, 12, "[P0] 增大批次"),
        ("③ Width=256",            256, 6,  "[P1] -20% 计算量"),
        ("④ Width=192",            192, 6,  "[P1] -40% 计算量"),
        ("⑤ W256+B12",             256, 12, "[P0+P1] 组合优化"),
    ]

    results = []
    warmup_runs = 0     # 跳过预热 (加快)
    bench_runs = 1      # 只跑1次 (快速出结果)

    for label, width, batch, desc in configs:
        print(f"\n--- [{label}] {desc} ---")
        print(f"    参数: width={width}, batch={batch}")

        try:
            engine = OCREngine(
                params=OCR_PARAMS,
                use_int8_rec=False,
                ort_threads=ORT_THREADS,
                rec_width=width,
                batch_num=batch,
            )
            engine.init_model()

            # Warmup: 让 JIT 编译和线程池稳定
            for _ in range(warmup_runs):
                engine.recognize(test_img)

            # Benchmark: 多次测量取平均
            rec_times = []
            total_times = []
            det_times = []
            last_result = None
            
            for run_idx in range(bench_runs):
                result = engine.recognize(test_img)
                if result:
                    rec_times.append(result.get('elapsed_rec', 0))
                    total_times.append(result.get('elapsed', 0))
                    det_times.append(result.get('elapsed_det', 0))
                    last_result = result

            if rec_times:
                avg_rec = sum(rec_times) / len(rec_times)
                avg_total = sum(total_times) / len(total_times)
                avg_det = sum(det_times) / len(det_times)
                n_texts = last_result['count'] if last_result else 0
                avg_score = last_result['avg_score'] if last_result else 0

                results.append({
                    'label': label,
                    'width': width,
                    'batch': batch,
                    'avg_det_ms': avg_det * 1000,
                    'avg_rec_ms': avg_rec * 1000,
                    'avg_total_ms': avg_total * 1000,
                    'n_texts': n_texts,
                    'avg_score': avg_score,
                })

                print(f"    结果: Det={avg_det*1000:.1f}ms | "
                      f"Rec={avg_rec*1000:.1f}ms | "
                      f"Total={avg_total*1000:.1f}ms | "
                      f"文字数={n_texts} | 均分={avg_score:.3f}")
            else:
                print("    [无结果]")

            # 释放模型资源
            del engine._ocr
            del engine

        except Exception as e:
            print(f"    [错误] {e}")
            continue

    # ═══════════════════════════════════════
    #  汇总表格 + 分析
    # ═══════════════════════════════════════
    if not results:
        print("\n[BENCH] 所有配置均失败，无法生成报告")
        return

    print("\n" + "=" * 90)
    print("  [BENCH] 基准测试结果汇总")
    print("=" * 90)
    print(f"  {'配置':<26s} {'宽':>3s} {'Bat':>4s} "
          f"{'Det(ms)':>8s} {'Rec(ms)':>9s} {'Tot(ms)':>9s} "
          f"{'文字':>5s} {'均分':>6s} {'Rec提速':>7s}")
    print("  " + "-" * 88)

    baseline_rec = None
    baseline_label = ""
    for r in results:
        if r['label'].startswith('①'):
            baseline_rec = r['avg_rec_ms']
            baseline_label = r['label']
            speed_ratio = "1.00x"
        elif baseline_rec and baseline_rec > 0:
            ratio = baseline_rec / r['avg_rec_ms']
            speed_ratio = f"{ratio:.2f}x"
        else:
            speed_ratio = "N/A"

        print(f"  {r['label']:<26s} {r['width']:>3d} {r['batch']:>4d} "
              f"{r['avg_det_ms']:>8.1f} {r['avg_rec_ms']:>9.1f} {r['avg_total_ms']:>9.1f} "
              f"{r['n_texts']:>5d} {r['avg_score']:>6.3f} {speed_ratio:>7s}")

    print("  " + "-" * 88)

    # ═════ 分析建议 ═════
    print("\n  [分析]")
    
    if len(results) >= 2:
        # 最快 Rec
        best = min(results, key=lambda x: x['avg_rec_ms'])
        print(f"     ★ 最快 Rec: [{best['label']}] ({best['avg_rec_ms']:.1f}ms)")
        
        if baseline_rec:
            improvement = (baseline_rec - best['avg_rec_ms']) / baseline_rec * 100
            print(f"     ★ 相对 Baseline 提升: {improvement:+.1f}%")

        # ── 宽度对比 (固定 batch=6) ──
        w320 = next((r for r in results if r['width'] == 320 and r['batch'] == 6), None)
        w256 = next((r for r in results if r['width'] == 256 and r['batch'] == 6), None)
        w192 = next((r for r in results if r['width'] == 192 and r['batch'] == 6), None)

        if w320 and w256:
            text_diff = abs(w256['n_texts'] - w320['n_texts'])
            score_diff = w256['avg_score'] - w320['avg_score']
            rec_speedup = w320['avg_rec_ms'] / w256['avg_rec_ms'] if w256['avg_rec_ms'] > 0 else 0
            print(f"\n     ── [P1] 宽度降低影响 (batch=6 固定):")
            print(f"       320→256: Rec {w320['avg_rec_ms']:.1f}→{w256['avg_rec_ms']:.1f}ms "
                  f"(×{rec_speedup:.2f}), "
                  f"文字 {w320['n_texts']}→{w256['n_texts']}(Δ{text_diff}), "
                  f"均分 {w320['avg_score']:.3f}→{w256['avg_score']:.3f}(Δ{score_diff:+.3f})")
            
            if text_diff == 0 and score_diff > -0.05:
                print(f"       [OK] 推荐 Width=256: 精度几乎无损，速度提升明显")
            elif text_diff <= 1 and score_diff > -0.10:
                print(f"       [!] 可接受 Width=256: 轻微精度损失换速度")

        if w320 and w192:
            text_diff = abs(w192['n_texts'] - w320['n_texts'])
            score_diff = w192['avg_score'] - w320['avg_score']
            rec_speedup = w320['avg_rec_ms'] / w192['avg_rec_ms'] if w192['avg_rec_ms'] > 0 else 0
            print(f"       320→192: Rec {w320['avg_rec_ms']:.1f}→{w192['avg_rec_ms']:.1f}ms "
                  f"(×{rec_speedup:.2f}), "
                  f"文字 {w320['n_texts']}→{w192['n_texts']}(Δ{text_diff}), "
                  f"均分 {w320['avg_score']:.3f}→{w192['avg_score']:.3f}(Δ{score_diff:+.3f})")

        # ── 批次对比 (固定 width=320) ──
        b6 = next((r for r in results if r['width'] == 320 and r['batch'] == 6), None)
        b12 = next((r for r in results if r['width'] == 320 and r['batch'] == 12), None)
        b16 = next((r for r in results if r['width'] == 320 and r['batch'] == 16), None)

        if b6 and b12:
            rec_speedup = b6['avg_rec_ms'] / b12['avg_rec_ms'] if b12['avg_rec_ms'] > 0 else 0
            print(f"\n     ── [P0] 批次增大影响 (width=320 固定):")
            print(f"       6→12:   Rec {b6['avg_rec_ms']:.1f}→{b12['avg_rec_ms']:.1f}ms "
                  f"(×{rec_speedup:.2f})")
        
        if b6 and b16:
            rec_speedup = b6['avg_rec_ms'] / b16['avg_rec_ms'] if b16['avg_rec_ms'] > 0 else 0
            print(f"       6→16:   Rec {b6['avg_rec_ms']:.1f}→{b16['avg_rec_ms']:.1f}ms "
                  f"(×{rec_speedup:.2f})")

        # ── 组合优化推荐 ──
        combo = next((r for r in results if '组合' in r['label'] or 'W256+B' in r['label']), None)
        if combo and baseline_rec:
            combo_speedup = baseline_rec / combo['avg_rec_ms'] if combo['avg_rec_ms'] > 0 else 0
            print(f"\n     ── [P0+P1] 组合优化:")
            combo_label = combo['label']
            print(f"       [{combo_label}]: Rec {combo['avg_rec_ms']:.1f}ms "
                  f"(相对Baseline x{combo_speedup:.2f})")

    print("\n" + "=" * 90)
    print("  [结论] 将以下配置写入配置区即可生效:")
    print(f"     REC_IMAGE_WIDTH = ???   # 根据上方 Width 对比选择")
    print(f"     REC_BATCH_NUM     = ???   # 根据上方 Batch 对比选择")
    print(f"     ENABLE_BENCH_MODE = False # 关闭基准测试，回到正常模式")
    print("=" * 90)


def main():
    # ── Benchmark 模式 (Rec 加速测试) ──
    if ENABLE_BENCH_MODE:
        run_benchmark()
        return

    # 初始化 OCR 引擎
    engine = OCREngine(
        params=OCR_PARAMS,
        use_int8_rec=(USE_INT8_REC == 1),
        ort_threads=ORT_THREADS,
        rec_width=REC_IMAGE_WIDTH,
        batch_num=REC_BATCH_NUM,
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
