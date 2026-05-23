"""
================================================================================
  【搬家指南 — 拷贝这个文件时需要带的文件】
================================================================================

  本文件所有路径均为【相对路径】，基于本文件(__file__)所在目录。
  只需保持以下文件夹结构即可，无需修改任何代码：

  项目文件夹/
  |-- ocr_workflow_accelerated.py     <-- 【本文件】
  |-- pp-ocrv4_rapid_onnx/            <-- 【必须带】模型文件夹 (整个文件夹拷过去)
  |   |-- ch_PP-OCRv4_det_mobile.onnx      文字检测模型 (Det) ~4.6MB
  |   |-- ch_ppocr_mobile_v2.0_cls_mobile.onnx  方向分类模型 (Cls) ~1.5MB
  |   |-- ch_PP-OCRv4_rec_mobile.onnx        文字识别模型 (Rec, FP32) ~10MB 主用
  |   |-- ppocr_keys_v1.txt                  CTC字符字典 (6625字) 必须
  |   |-- ch_PP-OCRv4_rec_mobile_int8.onnx   Rec INT8量化版 (实验性, 不用管)
  |-- photos/                       (测试图片, 可选)

  模型来源: PaddleOCR v4 mobile 系列 (pp-ocrv4_rapid_onnx)
    - Det: 检测图片中所有文字区域 -> 输出文字框坐标
    - Cls: 判断每个文字框的方向(0/90/180/270度) -> 自动旋转正
    - Rec: 将旋转校正后的文字区域逐个解码成文本

  安装依赖: pip install onnxruntime opencv-python numpy pyclipper shapely scikit-learn
================================================================================

OCR 工作流 -- PC 加速版 (功能整合 + 原生 ONNXRuntime 直推加速)
====================================================================
主体: ocr_workflow_onnx_linux.py (完整功能)
加速: ocr_board_v3.py (原生 ORT 推理, 绕过 RapidOCR 封装开销)

用法: python ocr_workflow_accelerated.py
识别结果: 直接打印到终端
测试模式: 运行后自动用 TEST_IMAGE_PATHS 测试, 结果输出到 ocr_test_result.txt
★ 只需改下方配置区，然后运行即可 ★

========================================
【部署到不同环境时需修改的参数】
========================================

--- 必须修改（换机器/换模型时） ---
MODEL_DIR         → ONNX模型文件夹路径，更换模型或迁移时改为实际目录
DICT_PATH         → CTC字符字典路径，与Rec模型配套，换模型时同步更换

--- 常用配置 (每次使用可能需要调整) ---
MODE              → 运行模式: 1=图片识别  2=摄像头拍照+OCR
OCR_ENGINE        → OCR引擎: 1=云端优先(失败降级本地)  2=纯本地(不联网)
IMAGE_PATHS       → 模式1要识别的图片路径列表(支持多张)
ENABLE_PREPROCESS → 图像预处理: True开(模糊/光照不均时)  False关(清晰扫描件)

--- 功能开关 ---
FILTER_KEYWORDS   → 屏蔽关键词列表(含这些词的结果不显示,如OSD水印)
HIDE_DRUG_META    → 药品说明书章节屏蔽: 0=显示全部  1=屏蔽法定信息章节
HIDE_SECTION_HEADERS → 屏蔽的章节标题列表(可自行增减)
SHOW_SCORES       → 置信度显示: 1=显示[0.95](调试)  0=隐藏(正式输出简洁)

--- 高级参数 (一般不需改) ---
OCR_PARAMS        → Det检测阈值/置信度阈值
ORT_INTRA_THREADS / ORT_INTER_THREADS → 推理线程数 (CPU核心数相关)
REC_BATCH_NUM / CLS_BATCH_NUM → 批处理大小 (越大越快但吃内存)

--- OCR模块代码区域标记（搜索可快速定位）---
[OCR模块-导入]      → 导入语句，换非ONNX模型需替换
[OCR模块-配置]      → MODEL_DIR / DICT_PATH 路径等全局参数
[OCR模块-模型加载]  → OCREngineAccelerated.init_model() 方法
[OCR模块-预处理]    → _adaptive_brightness_fix() / preprocess_image() 函数
[OCR模块-检测]      → det_preprocess() / det_postprocess() 函数
[OCR模块-分类]      → cls_preprocess() / cls_postprocess() 函数
[OCR模块-识别]      → rec_preprocess() / rec_postprocess() / ctc_decode() 函数
[OCR模块-排序]      → sort_boxes_by_layout() KMeans双栏排序函数
[OCR模块-过滤]      → should_filter_text() 关键词/章节过滤函数
[OCR模块-核心方法]  → OCREngineAccelerated.recognize() 完整流程入口

--- 外部接口（供其他py文件调用）---
[OCR接口-初始化]   → init_ocr() 全局单例预加载模型
[OCR接口-识别]     → recognize_image(path) 标准对外接口(返回纯文本)
[OCR接口-完整结果] → recognize_image_full(path) 返回结构化字典(含坐标/置信度/耗时)
[OCR接口-批量]     → batch_recognize(paths) 批量识别+可选写txt
[OCR接口-回调]     → set_ocr_callback(fn) 设置识别完成回调函数
[OCR接口-状态查询] → get_ocr_status() 查询引擎状态

--- 加速要点 ---
  1. 原生 ONNXRuntime Session 直推 (省去 RapidOCR 封装层开销)
  2. CTC 字典预加载 (模块导入时即读入, 零推理时 I/O)
  3. ORT SessionOptions: intra/inter_op_num_threads + 图优化全开
  4. Rec/Cls 可调大 batch (批量推理提升吞吐)
"""

import os
import sys
import time
import re
import math
import cv2
import numpy as np
import base64

# ============================================================
#  ★ 预加载重型依赖 + 字典 (加速: 避免推理时首次 import 开销) ★
# ============================================================
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import pyclipper
    from shapely.geometry import Polygon as ShapelyPolygon
    _has_pyclipper = True
except ImportError:
    _has_pyclipper = False

try:
    import requests as _requests
    _has_requests = True
except ImportError:
    _has_requests = False

# ★ CTC 字典预加载 (模块导入时即读取) ★
_PRELOAD_DICT = None
_PRELOAD_DICT_SIZE = 0
try:
    _DICT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'pp-ocrv4_rapid_onnx', 'ppocr_keys_v1.txt'
    )
    if os.path.exists(_DICT_PATH):
        with open(_DICT_PATH, 'r', encoding='utf-8') as _f:
            _PRELOAD_DICT = [line.strip() for line in _f if line.strip()]
        _PRELOAD_DICT.insert(0, 'blank')
        _PRELOAD_DICT.append(' ')
        while len(_PRELOAD_DICT) < 6625:
            _PRELOAD_DICT.append('')
        _PRELOAD_DICT_SIZE = len(_PRELOAD_DICT)
except Exception:
    pass


# ============================================================
#  ★ 配置区（改这里就行）★★★
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'photos')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'pp-ocrv4_rapid_onnx')

# ── 模式选择 ──
MODE = 1   # 1=图片识别    2=摄像头拍照+识别

# ── 模式1：填要识别的图片路径 ──
IMAGE_PATHS = [
    "photos/photo_4.jpg",
    # "photos/photo_6.jpg",
]



# ── 预处理开关（模糊/光照不均时开，清晰扫描件可关）──
ENABLE_PREPROCESS = True

# ── 屏蔽关键词：包含这些词的结果不显示 ──
FILTER_KEYWORDS = [
    "MJPG", "fps", "CPU:", "RAM:", "App:", "Photos:", "Photas:",
    "SPACE:", "shot", "quit",
]

# ── 屏蔽药品说明书元信息章节 ──
HIDE_DRUG_META = 1   # 0=显示全部  1=屏蔽以下章节
HIDE_SECTION_HEADERS = [
    "【执行标准】", "【批准文号】", "【说明书修订日期】",
    "【上市许可持有人】", "【生产企业】", "【包装】",
    "【境内联系机构】", "【药品上市许可持有人】",
]

# ── 显示置信度 ──
SHOW_SCORES = 0

# ── 摄像头模式: 拍照后自动退出 ──
CAMERA_AUTO_SHOT = 1           # 1=拍完自动退出  0=手动Q退出

# ── OCR 检测参数（重要！已调优勿动）──
OCR_PARAMS = {
    "Det.thresh": 0.20,
    "Det.box_thresh": 0.35,
    "Global.text_score": 0.4,
}

# ── 模型参数 (来自 ocr_board_v3.py 加速配置) ──
DET_ONNX = 'ch_PP-OCRv4_det_mobile.onnx'
CLS_ONNX = 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'
REC_ONNX = 'ch_PP-OCRv4_rec_mobile.onnx'       # FP32 推荐
# REC_ONNX = 'ch_PP-OCRv4_rec_mobile_int8.onnx' # INT8 实验性

DICT_PATH = os.path.join(MODEL_DIR, 'ppocr_keys_v1.txt')

# ── Det 参数 ──
DET_LIMIT_SIDE_LEN = 736
DET_LIMIT_TYPE = 'min'
DET_THRESH = 0.20
DET_BOX_THRESH = 0.35
DET_UNCLIP_RATIO = 1.6
DET_MAX_CANDIDATES = 1000
DET_USE_DILATION = True

# ── Cls 参数 ──
CLS_IMAGE_SHAPE = [3, 48, 192]
CLS_BATCH_NUM = 16      # ★ 加速: 增大批次
CLS_THRESH = 0.9

# ── Rec 参数 ──
REC_IMAGE_SHAPE = [3, 48, 320]
REC_BATCH_NUM = 16      # ★ 加速: 增大批次

# ── 全局参数 ──
TEXT_SCORE_THRESH = 0.4

# ── ORT 线程数 (加速关键) ──
ORT_INTRA_THREADS = 4   # 算子内并行线程数
ORT_INTER_THREADS = 2   # 算子间并行线程数

# ── 云端OCR配置 (OCR_ENGINE=1时云端优先，失败自动降级到本地) ──
CLOUD_OCR_URL = "https://z3c479c7k918obg7.aistudio-app.com/ocr"
CLOUD_OCR_TOKEN = "71bf88a8876217646a15669adb25fa0396d3e709"
CLOUD_OCR_TIMEOUT = 15  # API请求超时(秒)
OCR_ENGINE = 1           # 1=云端优先(失败降级本地)  2=纯本地(不联网)


# ============================================================
#  云端OCR (优先模式，失败自动降级本地)
# ============================================================

def cloud_ocr_recognize(image_path):
    """
    云端OCR识别 — 调用百度AI Studio OCR-API
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        dict: 识别结果 (与本地OCR格式兼容) {
            'texts': [str, ...],
            'scores': [float, ...],
            'boxes': [],
            'count': int,
            'avg_score': float,
            'elapsed': float,
            'elapsed_encode': float,
            'elapsed_request': float,
            'source': 'cloud',
        }
        失败时返回 None
    """
    if not _has_requests:
        print("  [云端OCR] requests库未安装，无法使用云端OCR")
        return None

    # 读取图片 + base64编码
    t0 = time.perf_counter()
    try:
        with open(image_path, "rb") as f:
            file_bytes = f.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")
    except Exception as e:
        print(f"  [云端OCR] 读取图片失败: {e}")
        return None
    t_encode = time.perf_counter() - t0

    # 发送API请求
    headers = {
        "Authorization": f"token {CLOUD_OCR_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "file": file_data,
        "fileType": 1,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }

    t1 = time.perf_counter()
    try:
        response = _requests.post(
            CLOUD_OCR_URL, json=payload, headers=headers,
            timeout=CLOUD_OCR_TIMEOUT
        )
        t_request = time.perf_counter() - t1
    except _requests.exceptions.ConnectionError:
        print("  [云端OCR] 网络连接失败！请检查网络是否正常")
        print("  [云端OCR] → 自动切换到本地OCR模式")
        return None
    except _requests.exceptions.Timeout:
        print(f"  [云端OCR] 请求超时({CLOUD_OCR_TIMEOUT}s)！网络可能不稳定")
        print("  [云端OCR] → 自动切换到本地OCR模式")
        return None
    except Exception as e:
        print(f"  [云端OCR] 请求异常: {e}")
        print("  [云端OCR] → 自动切换到本地OCR模式")
        return None

    if response.status_code != 200:
        print(f"  [云端OCR] 服务端返回错误: HTTP {response.status_code}")
        print(f"  [云端OCR] → 自动切换到本地OCR模式")
        return None

    # 解析结果
    try:
        resp_json = response.json()
        result = resp_json["result"]
        ocr_results = result.get("ocrResults", [])
    except Exception as e:
        print(f"  [云端OCR] 解析响应失败: {e}")
        print("  [云端OCR] → 自动切换到本地OCR模式")
        return None

    # 提取文字和分数
    all_texts = []
    all_scores = []
    for res in ocr_results:
        raw = res.get("prunedResult", "")
        if isinstance(raw, dict):
            rec_texts = raw.get("rec_texts", [])
            rec_scores = raw.get("rec_scores", [])
            for i, txt in enumerate(rec_texts):
                all_texts.append(txt)
                score = rec_scores[i] if i < len(rec_scores) else 0.95
                all_scores.append(score)
        elif isinstance(raw, str) and raw:
            all_texts.append(raw)
            all_scores.append(0.95)

    total_elapsed = time.perf_counter() - t0

    return {
        'texts': all_texts,
        'scores': all_scores,
        'boxes': [],
        'count': len(all_texts),
        'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'elapsed': total_elapsed,
        'elapsed_encode': t_encode,
        'elapsed_request': t_request,
        'source': 'cloud',
    }


# ============================================================
#  统一识别入口 (云端优先 + 本地降级)
# ============================================================

def recognize_with_fallback(engine, image_path):
    """
    统一识别入口: 云端OCR优先，失败自动降级到本地OCR
    
    Args:
        engine: 本地OCR引擎实例 (OCREngineAccelerated)
        image_path: 图片路径
        
    Returns:
        dict: 识别结果 (含 'source' 字段标识来源: 'cloud' 或 'local')
    """
    # 云端优先
    if OCR_ENGINE == 1:
        cloud_result = cloud_ocr_recognize(image_path)
        if cloud_result is not None and cloud_result.get('texts'):
            return cloud_result
        # 云端失败，降级到本地
        print("  [降级] 使用本地OCR进行识别...")

    # 本地OCR
    result = engine.recognize(image_path)
    if result is not None:
        result['source'] = 'local'
    return result


# ============================================================
#  图像预处理 (来自 linux 版, 含自适应亮度校正)
# ============================================================

def _adaptive_brightness_fix(img):
    """自适应亮度校正 (仅严重过曝时触发)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())

    needs_fix = mean_val > 180 or std_val < 35
    if not needs_fix:
        return img

    if mean_val > 170:
        bright_score = max((mean_val - 180) / 70, 0)
        low_contrast = max((40 - std_val) / 25, 0) if std_val < 40 else 0
        severity = max(bright_score, low_contrast * 0.6)
        severity = min(severity, 1.0)
        gamma = 1.2 + severity * 0.45
        table = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255)
                          for i in range(256)], dtype='uint8')
        img = cv2.LUT(img, table)
    return img


def preprocess_image(img):
    """OCR 专用图像预处理: 锐化 + CLAHE 对比度增强"""
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 1.45, blurred, -0.45, 0)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return result


def load_image(path):
    """读取图片（支持中文路径）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ============================================================
#  摄像头模块 (来自 ocr_workflow_onnx_linux.py)
# ============================================================

def find_best_camera():
    """快速探测摄像头设备（轻量版，不读帧不设参）"""
    for priority_idx in [2, 0, 1, 3, 4]:
        cap = cv2.VideoCapture(priority_idx, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.release()
            return priority_idx
    return None


def setup_camera(cap):
    """设置摄像头参数 - 分辨率/格式/MJPG/帧率"""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)


def _sharpness_score(frame):
    """计算图像清晰度评分 (Laplacian 方差)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _smart_shutter(cap, wait_ms=350, burst=5):
    """智能快门: 解决手持拍照拖影问题"""
    start = time.time()
    while (time.time() - start) * 1000 < wait_ms:
        cap.read()

    best_frame = None
    best_score = -1
    for _ in range(burst):
        ret, frame = cap.read()
        if ret and frame is not None:
            score = _sharpness_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

    if best_frame is None:
        ret, best_frame = cap.read()
        if not ret or best_frame is None:
            return None
    return best_frame


def camera_mode(ocr_engine):
    """摄像头模式：实时预览 + 智能快门(防抖) + 自动识别"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 50)
    print("  USB Camera - OCR Mode")
    print("  Smart Shutter: ON (anti-shake, 5-frame burst)")
    print("=" * 50)

    cam_idx = find_best_camera()
    if cam_idx is None:
        print("ERROR: 未找到摄像头!")
        print("       请确认:")
        print("         1. USB 摄像头已插入 (ls /dev/video*)")
        print("         2. 使用 root 权限运行 (sudo python3 ...)")
        return

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: 无法打开摄像头!")
        return

    setup_camera(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win = 'Camera'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    def _next_id():
        max_id = 0
        for f in os.listdir(SAVE_DIR):
            import re
            m = re.match(r'photo_(\d+)\.', f)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    photo_count = _next_id()
    fps_list, t_last, fps_show = [], time.time(), 0

    has_psutil = True
    try:
        import psutil
    except:
        has_psutil = False

    if CAMERA_AUTO_SHOT:
        print(f"[OK] {w}x{h} | SPACE=拍照(识别后自动退出)  Q=取消\n")
    else:
        print(f"[OK] {w}x{h} | SPACE=拍照(防抖)  Q=退出\n")
    print("提示: 按空格后等待~350ms稳定, 连拍5帧选最清晰\n")

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

        cpu_v, mem_v, proc_v = 0, 0, 0
        if has_psutil:
            try:
                cpu_v = psutil.cpu_percent()
                mem_v = psutil.virtual_memory().percent
                proc_v = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            except:
                pass

        info = [
            f"{w}x{h}  {fps_show}fps",
            f"CPU:{cpu_v:.0f}%  RAM:{mem_v:.0f}%",
            f"Photos: {photo_count}",
            "SPACE:shot  Q:quit",
        ]
        y = 28
        for line in info:
            cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 120), 2)
            y += 28

        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord(' '):
            print("  [SHUTTER] capturing...", end='', flush=True)
            photo_frame = _smart_shutter(cap, wait_ms=350, burst=5)

            if photo_frame is not None:
                photo_count += 1
                path = os.path.join(SAVE_DIR, f'photo_{photo_count}.jpg')
                ok, buf = cv2.imencode('.jpg', photo_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                if ok:
                    with open(path, 'wb') as f:
                        f.write(buf)
                    sz = os.path.getsize(path) / 1024
                    score = _sharpness_score(photo_frame)
                    print(f" done! (sharpness={score:.0f}, {sz:.0f}KB)")

                    # 立即 OCR 识别
                    print(f"\n{'='*60}")
                    print(f"  [已保存] photo_{photo_count}.jpg ({sz:.0f}KB)")
                    result = ocr_engine.recognize(photo_frame)
                    if result:
                        print_result(result, path)

                    if CAMERA_AUTO_SHOT:
                        print("\n[AUTO] 识别完成，自动退出...")
                        break
            else:
                print(" FAILED")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {photo_count} photos saved.")


def load_dict(dict_path):
    """加载 CTC 解码字典"""
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f if line.strip()]
    chars.insert(0, 'blank')
    chars.append(' ')
    while len(chars) < 6625:
        chars.append('')
    return chars, len(chars)


# ============================================================
#  文本框排序 (linux 版 IQR 改进版)
# ============================================================

def sort_boxes_by_layout(boxes, texts, scores, mode="auto"):
    """按阅读顺序重新排序文本框 (IQR 改进双栏检测)"""
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
        # ── KMeans(k=2) 聚类分栏 (对OSD水印等离群值鲁棒) ──
        #   旧版 IQR 最大间隙法的问题:
        #     OSD水印(x=60~160) 与正文首行(x=547) 之间产生384px假间隙
        #     -> split_x=354 (仅图片宽度18%), 左栏只有4个水印垃圾文本
        #     -> 所有正文混入右栏 -> 按Y交错排列 = 读串列
        #   KMeans 方案: 按 x 坐标聚成2簇, 簇中心自然对应左右列
        try:
            from sklearn.cluster import KMeans
            x_2d = x_centers.reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(x_2d)

            # 确保 cluster_0 是左列(中心x较小), cluster_1 是右列
            centers_sorted = np.argsort(kmeans.cluster_centers_.flatten())
            left_label = centers_sorted[0]
            right_label = centers_sorted[1]

            left_idx = np.where(labels == left_label)[0]
            right_idx = np.where(labels == right_label)[0]

            split_x = float(kmeans.cluster_centers_[left_label])
        except ImportError:
            split_x = (x_centers.max() + x_centers.min()) / 2
            left_mask = x_centers <= split_x
            right_mask = x_centers > split_x
            left_idx = np.where(left_mask)[0]
            right_idx = np.where(right_mask)[0]

        left_sorted = left_idx[np.argsort(y_centers[left_idx])]
        right_sorted = right_idx[np.argsort(y_centers[right_idx])]
        indices = np.concatenate([left_sorted, right_sorted])
    else:
        indices = np.arange(n)

    return (
        [boxes[i] for i in indices],
        [texts[i] for i in indices],
        [scores[i] for i in indices],
    )


# ============================================================
#  Det 预处理 / 后处理 (来自 v3 加速版)
# ============================================================

def det_preprocess(img, limit_side_len=736, limit_type='min', pad_to_square=False):
    """文本检测预处理 — 对齐 RapidOCR DetPreProcess"""
    h, w = img.shape[:2]

    if limit_type == 'max':
        if max(h, w) > limit_side_len:
            ratio = float(limit_side_len) / h if h > w else float(limit_side_len) / w
        else:
            ratio = 1.0
    else:
        if min(h, w) < limit_side_len:
            ratio = float(limit_side_len) / h if h < w else float(limit_side_len) / w
        else:
            ratio = 1.0

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    if resize_h <= 0 or resize_w <= 0:
        return None, None

    resized = cv2.resize(img, (resize_w, resize_h))

    if pad_to_square:
        sq = max(resize_h, resize_w)
        if sq == 0:
            return None, None
        padded = np.zeros((sq, sq, 3), dtype=np.uint8)
        padded[:resize_h, :resize_w] = resized
        work_img = padded
    else:
        work_img = resized

    normalized = work_img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (normalized - mean) / std
    normalized = normalized.transpose((2, 0, 1))
    normalized = np.expand_dims(normalized, axis=0).astype(np.float32)

    if pad_to_square:
        return normalized, (sq, sq, ratio, h, w, resize_h, resize_w)
    else:
        return normalized, (resize_h, resize_w, ratio, h, w)


def det_postprocess(det_result, original_shape, resize_info,
                    thresh=0.30, box_thresh=0.50,
                    unclip_ratio=2.0, max_candidates=1000,
                    use_dilation=False):
    """DB 后处理 — 对齐 RapidOCR DBPostProcess"""
    if isinstance(det_result, dict):
        key = list(det_result.keys())[0]
        prob_data = np.array(det_result[key])
    else:
        prob_data = np.array(det_result)

    if prob_data.ndim == 4:
        prob_map = prob_data[0, 0]
    elif prob_data.ndim == 3:
        prob_map = prob_data[0]
    elif prob_data.ndim == 2:
        prob_map = prob_data
    else:
        raise ValueError(f"Unexpected det output shape: {prob_data.shape}")

    is_dlc_pad = (len(resize_info) == 7)
    if is_dlc_pad:
        sq, sq2, ratio, orig_h, orig_w, act_h, act_w = resize_info
        out_h, out_w = prob_map.shape[:2]
        if (out_h, out_w) != (sq, sq):
            prob_map = cv2.resize(prob_map, (sq, sq), interpolation=cv2.INTER_LINEAR)
        resize_h, resize_w = sq, sq
    else:
        resize_h, resize_w, ratio, orig_h, orig_w = resize_info

    min_size = 3
    segmentation = prob_map > thresh

    mask = segmentation.copy()
    if use_dilation:
        kernel = np.array([[1, 1], [1, 1]], np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    num_contours = min(len(contours), max_candidates)
    boxes, scores = [], []

    for ci in range(num_contours):
        contour = contours[ci]
        points, sside = _get_mini_boxes(contour)
        if sside < min_size:
            continue
        score = _box_score_fast(prob_map, points.reshape(-1, 2))
        if score < box_thresh:
            continue
        try:
            expanded = _unclip_pyclipper(points, unclip_ratio)
        except Exception:
            continue
        box, sside = _get_mini_boxes(expanded)
        if sside < min_size + 2:
            continue
        if is_dlc_pad:
            box[:, 0] = np.clip(np.round(box[:, 0] / act_w * orig_w), 0, orig_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / act_h * orig_h), 0, orig_h)
        else:
            box[:, 0] = np.clip(np.round(box[:, 0] / resize_w * orig_w), 0, orig_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / resize_h * orig_h), 0, orig_h)
        boxes.append(box)
        scores.append(score)

    boxes, scores = _filter_det_res(boxes, scores, orig_h, orig_w)
    return boxes, scores


def _get_mini_boxes(contour):
    """最小外接矩形 + 4点顺时针排列"""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0

    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    box = np.array([
        points[index_1], points[index_2],
        points[index_3], points[index_4],
    ], dtype=np.float32)
    return box, min(bounding_box[1])


def _box_score_fast(bitmap, _box):
    """快速计算框内概率均值"""
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box_local = box.copy()
    box_local[:, 0] -= xmin
    box_local[:, 1] -= ymin
    cv2.fillPoly(mask, box_local.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask=mask)[0])


def _unclip_pyclipper(box, unclip_ratio):
    """pyclipper 多边形扩展"""
    if not _has_pyclipper:
        raise ImportError("pyclipper/shapely not available")
    poly = ShapelyPolygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return np.array(expanded[0], dtype=np.float32).reshape((-1, 1, 2))


def _filter_det_res(dt_boxes, scores, img_height, img_width):
    """最终候选框过滤"""
    dt_boxes_new = []
    new_scores = []
    for box, score in zip(dt_boxes, scores):
        box = _order_points_clockwise(box.astype(np.float32))
        for pno in range(4):
            box[pno, 0] = float(min(max(int(round(box[pno, 0])), 0), img_width - 1))
            box[pno, 1] = float(min(max(int(round(box[pno, 1])), 0), img_height - 1))
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
        new_scores.append(score)
    return dt_boxes_new, new_scores


def _order_points_clockwise(pts):
    """将4个点按顺时针排列 (tl, tr, br, bl)"""
    pts = np.array(pts, dtype=np.float32)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tl, bl = left_most
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_rotate_crop_image(img, box):
    """裁剪并矫正旋转文本区域 — 对齐 RapidOCR process_img.py"""
    points = box.astype(np.float32)
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))

    pts_std = np.array([
        [0, 0], [img_crop_width, 0],
        [img_crop_width, img_crop_height], [0, img_crop_height],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_h, dst_img_w = dst_img.shape[:2]
    if dst_img_h * 1.0 / dst_img_w >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


# ============================================================
#  Cls 预处理 / 后处理
# ============================================================

def cls_preprocess(images, target_shape=(48, 192)):
    """方向分类预处理: resize -> normalize -> CHW -> batch"""
    batch = []
    th, tw = target_shape
    for img in images:
        resized = cv2.resize(img, (tw, th))
        normed = resized.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        normed = (normed - mean) / std
        normed = normed.transpose((2, 0, 1))
        batch.append(normed)
    return np.stack(batch, axis=0)


def cls_postprocess(output, thresh=0.9):
    """方向分类后处理: 返回是否需要旋转180度"""
    preds = np.array(output['save_infer_model/scale_0.tmp_1'])
    if isinstance(preds, list):
        preds = np.array(preds[0])
    angles = []
    scores_list = []
    for pred in preds:
        idx = np.argmax(pred)
        score = float(pred[idx])
        angles.append(idx)
        scores_list.append(score)
    return angles, scores_list


# ============================================================
#  Rec 预处理 / 后处理 (CTC 解码)
# ============================================================

def rec_preprocess(images, target_shape=(48, 320)):
    """文字识别预处理 — 对齐 RapidOCR resize_norm_img"""
    imgC, imgH, imgW = 3, target_shape[0], target_shape[1]
    max_wh_ratio = imgW / imgH

    for img in images:
        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)

    actual_imgW = int(imgH * max_wh_ratio)
    batch = []
    for img in images:
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > actual_imgW:
            resized_w = actual_imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype(np.float32)
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, actual_imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        batch.append(padding_im)
    return np.stack(batch, axis=0)


def ctc_decode_greedy(probs, dict_chars):
    """CTC Greedy 解码 — 对齐 RapidOCR CTCLabelDecode"""
    texts = []
    text_scores = []

    if probs.ndim == 3:
        batch_size = probs.shape[0]
    else:
        probs = probs[np.newaxis]
        batch_size = 1

    for b in range(batch_size):
        pred_idx = np.argmax(probs[b], axis=1)
        pred_prob = np.max(probs[b], axis=1)
        selection = np.ones(len(pred_idx), dtype=bool)
        selection[1:] = pred_idx[1:] != pred_idx[:-1]
        selection &= (pred_idx != 0)

        if selection.any():
            conf_list = pred_prob[selection].tolist()
            conf_list = [round(c, 5) for c in conf_list]
            selected_idx = pred_idx[selection]
            valid_mask = selected_idx < len(dict_chars)
            if not valid_mask.all():
                selected_idx = selected_idx[valid_mask]
                conf_list = [c for c, v in zip(conf_list, valid_mask) if v]
            char_list = [dict_chars[idx] for idx in selected_idx if dict_chars[idx]]
            if len(char_list) != len([dict_chars[idx] for idx in selected_idx]):
                _valid_chars = [(dict_chars[idx], c) for idx, c in zip(selected_idx, conf_list)
                                if idx < len(dict_chars) and dict_chars[idx]]
                if _valid_chars:
                    char_list = [c[0] for c in _valid_chars]
                    conf_list = [c[1] for c in _valid_chars]
            text = ''.join(char_list)
            avg_score = float(np.mean(conf_list)) if conf_list else 0.0
        else:
            text = ''
            avg_score = 0.0
        texts.append(text)
        text_scores.append(avg_score)

    return texts, text_scores


# ============================================================
#  ★ 核心: OCR 引擎 (原生 ONNXRuntime 直推加速版) ★
# ============================================================

class OCREngineAccelerated:
    """
    PC 加速版 OCR 引擎
    
    加速策略 (vs RapidOCR 封装版):
      1. 原生 ORT Session 直推 (消除 RapidOCR 封装层开销)
      2. CTC 字典模块级预加载 (零推理时 I/O)
      3. SessionOptions: graph_optimization_level=ALL + 多线程
      4. 大 batch 批量推理 (Cls=16, Rec=16)
    
    功能保留 (来自 linux 版):
      - 自适应亮度校正 + 条件预处理跳过
      - IQR 改进双栏排序
      - 关键词过滤 + 药品章节屏蔽
      - 完整结果格式化输出
    """

    def __init__(self):
        self.det_session = None
        self.cls_session = None
        self.rec_session = None
        self.dict_chars = None
        self.dict_size = 0
        self._load_time = 0

    def init_model(self):
        """加载所有模型 (原生 ONNXRuntime)"""
        if ort is None:
            raise ImportError("需要安装 onnxruntime: pip install onnxruntime")

        t0 = time.perf_counter()
        print("=" * 55)
        print("  PC Accelerated OCR Engine - Loading Models")
        print("  (Native ONNXRuntime Direct Inference)")
        print("=" * 55)
        print(f"  Det: {DET_ONNX}")
        print(f"  Cls: {CLS_ONNX}")
        print(f"  Rec: {REC_ONNX}")

        # ★ 使用预加载的 CTC 字典 ★
        if _PRELOAD_DICT is not None:
            self.dict_chars = _PRELOAD_DICT[:]
            self.dict_size = _PRELOAD_DICT_SIZE
            print(f"  Dict: preloaded ({self.dict_size} chars)")
        else:
            self.dict_chars, self.dict_size = load_dict(DICT_PATH)
            print(f"  Dict: loaded from file ({self.dict_size} chars)")

        det_path = os.path.join(MODEL_DIR, DET_ONNX)
        cls_path = os.path.join(MODEL_DIR, CLS_ONNX)
        rec_path = os.path.join(MODEL_DIR, REC_ONNX)

        # ★ 公共 SessionOptions (图优化全开 + 多线程) ★
        def _make_session_options():
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = ORT_INTRA_THREADS
            so.inter_op_num_threads = ORT_INTER_THREADS
            return so

        # Det
        print("  Loading Det...")
        self.det_session = ort.InferenceSession(det_path, _make_session_options())
        print("  [OK] Det loaded")

        # Cls
        print("  Loading Cls...")
        self.cls_session = ort.InferenceSession(cls_path, _make_session_options())
        print("  [OK] Cls loaded")

        # Rec
        print("  Loading Rec...")
        self.rec_session = ort.InferenceSession(rec_path, _make_session_options())
        print("  [OK] Rec loaded")

        self._load_time = time.perf_counter() - t0
        print(f"\n  All models loaded! ({self._load_time:.3f}s)\n")

    def recognize(self, image_or_path):
        """
        执行完整 OCR 识别
        
        返回格式 (与 linux 版 OCREngine 完全一致):
          {
            'texts': [...], 'scores': [...], 'boxes': [...],
            'count': int, 'avg_score': float, 'elapsed': float,
            'elapsed_det': float, 'elapsed_cls': float, 'elapsed_rec': float,
          }
        """
        if self.det_session is None:
            raise RuntimeError("请先调用 init_model()")

        # 加载图片
        if isinstance(image_or_path, str):
            img = load_image(image_or_path)
        else:
            img = image_or_path
        if img is None:
            return None

        # ★ 预处理 (来自 linux 版: 自适应亮度 + 条件跳过) ★
        if ENABLE_PREPROCESS:
            img = _adaptive_brightness_fix(img)
            gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            m_check = float(gray_check.mean())
            s_check = float(gray_check.std())
            needs_enhance = m_check <= 155 or m_check >= 195 or s_check < 22
            if needs_enhance:
                img = preprocess_image(img)

        orig_h, orig_w = img.shape[:2]

        # ===== Stage 1: Text Detection =====
        t0 = time.perf_counter()
        det_input, resize_info = det_preprocess(
            img,
            limit_side_len=DET_LIMIT_SIDE_LEN,
            limit_type=DET_LIMIT_TYPE,
            pad_to_square=False,
        )
        if det_input is None:
            return {'texts': [], 'scores': [], 'boxes': [],
                    'count': 0, 'avg_score': 0, 'elapsed': 0,
                    'elapsed_det': 0, 'elapsed_cls': 0, 'elapsed_rec': 0}
        det_output = self._run_det(det_input)
        det_t = time.perf_counter() - t0

        # ===== Stage 2: Post-process Detection =====
        t1 = time.perf_counter()
        boxes, box_scores = det_postprocess(
            det_output, img.shape, resize_info,
            thresh=DET_THRESH,
            box_thresh=DET_BOX_THRESH,
            unclip_ratio=DET_UNCLIP_RATIO,
            max_candidates=DET_MAX_CANDIDATES,
            use_dilation=DET_USE_DILATION,
        )
        post_t = time.perf_counter() - t1

        if len(boxes) == 0:
            return {'texts': [], 'scores': [], 'boxes': [],
                    'count': 0, 'avg_score': 0,
                    'elapsed': det_t + post_t,
                    'elapsed_det': det_t, 'elapsed_cls': 0, 'elapsed_rec': 0}

        # ===== Stage 3: Crop & Classify Direction =====
        t2 = time.perf_counter()
        crops = [get_rotate_crop_image(img, box) for box in boxes]
        all_angles = []
        all_cls_scores = []
        batch_size = CLS_BATCH_NUM
        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start:start + batch_size]
            cls_input = cls_preprocess(batch_crops, tuple(CLS_IMAGE_SHAPE[1:]))
            cls_output = self._run_cls(cls_input)
            angles, cls_scores = cls_postprocess(cls_output, CLS_THRESH)
            all_angles.extend(angles)
            all_cls_scores.extend(cls_scores)
        cls_t = time.perf_counter() - t2

        # 根据角度翻转图片 (高置信度才执行)
        rotated_crops = []
        for crop, angle, cls_sc in zip(crops, all_angles, all_cls_scores):
            if angle == 1 and cls_sc >= CLS_THRESH:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
            rotated_crops.append(crop)

        # ===== Stage 4: Text Recognition (CTC) =====
        t3 = time.perf_counter()
        all_texts, all_text_scores = [], []
        batch_size = REC_BATCH_NUM
        for start in range(0, len(rotated_crops), batch_size):
            batch_crops = rotated_crops[start:start + batch_size]
            rec_input = rec_preprocess(batch_crops, tuple(REC_IMAGE_SHAPE[1:]))
            rec_output = self._run_rec(rec_input)
            texts, text_scores = ctc_decode_greedy(rec_output, self.dict_chars)
            all_texts.extend(texts)
            all_text_scores.extend(text_scores)
        rec_t = time.perf_counter() - t3

        total_t = det_t + post_t + cls_t + rec_t

        # 过滤低置信度结果
        valid_texts, valid_scores, valid_boxes = [], [], []
        for txt, sc, bs, box in zip(all_texts, all_text_scores, box_scores, boxes):
            if sc >= TEXT_SCORE_THRESH:
                valid_texts.append(txt)
                valid_scores.append(sc)
                valid_boxes.append(box)

        # 排序
        h, w = img.shape[:2]
        mode = "double" if w / h > 1.5 else "single"
        sorted_boxes, sorted_texts, sorted_scores = sort_boxes_by_layout(
            valid_boxes, valid_texts, valid_scores, mode=mode
        )

        return {
            'texts': sorted_texts,
            'scores': sorted_scores,
            'boxes': sorted_boxes,
            'count': len(sorted_texts),
            'avg_score': sum(sorted_scores) / len(sorted_scores) if sorted_scores else 0,
            'elapsed': total_t,
            'elapsed_det': det_t,
            'elapsed_cls': cls_t,
            'elapsed_rec': rec_t,
        }

    def _run_det(self, input_data):
        """Det 推理 (原生 ORT)"""
        input_feed = {'x': input_data.astype(np.float32)}
        output_names = ['sigmoid_0.tmp_0']
        out = self.det_session.run(output_names, input_feed)
        return {output_names[0]: out[0]}

    def _run_cls(self, input_data):
        """Cls 推理 (原生 ORT)"""
        output_names = ['save_infer_model/scale_0.tmp_1']
        input_feed = {'x': input_data.astype(np.float32)}
        out = self.cls_session.run(output_names, input_feed)
        return {output_names[0]: out[0]}

    def _run_rec(self, input_data):
        """Rec 推理 (原生 ORT)"""
        output_names = ['softmax_11.tmp_0']
        input_feed = {'x': input_data.astype(np.float32)}
        out = self.rec_session.run(output_names, input_feed)
        return out[0]


# ============================================================
#  结果输出 (完全对齐 linux 版)
# ============================================================

def print_result(result, source_name=""):
    """格式化打印 OCR 识别结果 (含关键词过滤+药品章节屏蔽)"""
    filtered = []
    for text, score in zip(result['texts'], result['scores']):
        if any(kw in text for kw in FILTER_KEYWORDS):
            continue
        filtered.append((text, score))

    if HIDE_DRUG_META:
        hidden = 0
        hiding = False
        output = []
        for text, score in filtered:
            is_section_header = any(
                text.strip().startswith(h) or h in text
                for h in HIDE_SECTION_HEADERS
            )
            is_any_section = '[' in text and ']' in text

            if is_section_header:
                hiding = True
                hidden += 1
                continue
            elif hiding and is_any_section:
                hiding = False
                output.append((text, score))
            elif not hiding:
                output.append((text, score))
            else:
                hidden += 1
        filtered = output
        _hidden_count = hidden
    else:
        _hidden_count = 0

    count = len(filtered)
    total = result['count']
    skipped = total - count - _hidden_count

    status_parts = [f"识别到 {total} 段文字"]
    source = result.get('source', 'local')
    source_tag = "[CLOUD]" if source == 'cloud' else "[LOCAL]"
    status_parts.append(f"[{source_tag}]")
    if skipped > 0:
        status_parts.append(f"过滤 {skipped} 条水印")
    if _hidden_count > 0:
        status_parts.append(f"屏蔽 {_hidden_count} 条元信息")
    status_parts.append(f"显示 {count} 段")
    status_parts.append(f"平均置信度 {result['avg_score']:.3f}")

    print(f"  +--- {' | '.join(status_parts)}")
    if count == 0:
        print(f"  |  (all filtered or no text)")
    else:
        print("  |")
        for text, score in filtered:
            if SHOW_SCORES:
                tag = "" if score >= 0.95 else f"  [{score:.2f}]"
            else:
                tag = ""
            print(f"  |  {text}{tag}")
    print("  |")
    source = result.get('source', 'local')
    if source == 'cloud':
        enc_t = result.get('elapsed_encode', 0)
        req_t = result.get('elapsed_request', 0)
        print(f"  +-- Time: 编码={enc_t:.2f}s + API请求={req_t:.2f}s = Total={result['elapsed']:.2f}s")
    else:
        print(f"  +-- Time: Det={result['elapsed_det']:.2f}s + "
              f"Cls={result['elapsed_cls']:.2f}s + Rec={result['elapsed_rec']:.2f}s"
              f" = Total={result['elapsed']:.2f}s")
    print()


# ============================================================
#  测试模式: 自动运行多张图片 + 输出 txt 报告
# ============================================================

def run_test_and_save_report(engine, test_paths, output_txt_path):
    """
    自动测试模式: 对每张图片执行 OCR 并将结果保存到 txt 文件
    
    输出内容:
      - 每张图的识别耗时、文字数、平均置信度、所有识别文本
      - 汇总统计: 总耗时、总文字数、整体准确率指标
    """
    lines = []
    def log(msg=''):
        lines.append(msg)
        print(msg)

    total_start = time.perf_counter()
    total_images = 0
    total_texts = 0
    total_score_sum = 0
    total_score_count = 0
    all_results = {}

    log("=" * 70)
    log("  OCR PC 加速版 - 自动测试报告")
    log("=" * 70)
    log(f"  测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  测试图片数: {len(test_paths)}")
    log(f"  加速配置: ORT原生直推 | intra={ORT_INTRA_THREADS}threads | "
        f"inter={ORT_INTER_THREADS}threads | RecBatch={REC_BATCH_NUM} | ClsBatch={CLS_BATCH_NUM}")
    log()

    for idx, path in enumerate(test_paths, 1):
        if not os.path.exists(path):
            log(f"[{idx}/{len(test_paths)}] 跳过 - 文件不存在: {path}")
            log()
            continue

        log("-" * 70)
        log(f"  [{idx}/{len(test_paths)}] 图片: {path}")

        result = engine.recognize(path)
        if result is None or result['count'] == 0:
            log(f"  结果: 未识别到文字")
            log(f"  耗时: {result['elapsed']:.3f}s" if result else "  耗时: N/A")
            all_results[path] = {'result': result, 'texts': []}
            log()
            continue

        total_images += 1

        # 过滤输出 (与 print_result 逻辑一致)
        filtered = []
        for text, score in zip(result['texts'], result['scores']):
            if any(kw in text for kw in FILTER_KEYWORDS):
                continue
            filtered.append((text, score))

        if HIDE_DRUG_META:
            hidden = 0
            hiding = False
            output = []
            for text, score in filtered:
                is_section_header = any(
                    text.strip().startswith(h) or h in text
                    for h in HIDE_SECTION_HEADERS
                )
                is_any_section = '\u3010' in text and '\u3011' in text
                if is_section_header:
                    hiding = True
                    hidden += 1
                    continue
                elif hiding and is_any_section:
                    hiding = False
                    output.append((text, score))
                elif not hiding:
                    output.append((text, score))
                else:
                    hidden += 1
            filtered = output

        n_texts = len(filtered)
        total_texts += n_texts

        # 统计分数
        if filtered:
            scores_only = [s for _, s in filtered]
            avg = sum(scores_only) / len(scores_only)
            total_score_sum += sum(scores_only)
            total_score_count += len(scores_only)
        else:
            avg = 0

        log(f"  识别文字数: {n_texts}")
        log(f"  平均置信度: {avg:.4f}")
        log(f"  耗时: Det={result['elapsed_det']:.3f}s + "
            f"Cls={result['elapsed_cls']:.3f}s + Rec={result['elapsed_rec']:.3f}s"
            f" = Total={result['elapsed']:.3f}s")
        log(f"  --- 识别内容 ---")
        for text, score in filtered:
            score_tag = f" [{score:.4f}]" if SHOW_SCORES else ""
            log(f"    {text}{score_tag}")

        all_results[path] = {'result': result, 'texts': [t for t, _ in filtered]}
        log()

    # 汇总
    total_elapsed = time.perf_counter() - total_start
    overall_avg_score = total_score_sum / total_score_count if total_score_count > 0 else 0

    log("=" * 70)
    log("  === 汇总统计 ===")
    log("=" * 70)
    log(f"  成功识别图片数: {total_images} / {len(test_paths)}")
    log(f"  总识别文字段数: {total_texts}")
    log(f"  整体平均置信度: {overall_avg_score:.4f}")
    log(f"  总耗时: {total_elapsed:.3f}s")
    if total_images > 0:
        log(f"  平均每张耗时: {total_elapsed / total_images:.3f}s")
    log(f"  模型加载耗时: {engine._load_time:.3f}s")
    log()

    # 写入文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    log(f"  报告已保存到: {output_txt_path}")
    return all_results


# ============================================================
#  外部衔接接口（供其他py文件调用，与 asr.py 接口风格一致）
# ============================================================

# 全局 OCR 引擎单例（用于外部访问，避免重复加载模型）
_global_engine = None

# 全局 OCR 结果回调函数（由外部文件如 main.py / link.py 设置）
# 调用方式: from ocr_workflow_accelerated import set_ocr_callback; set_ocr_callback(your_function)
_ocr_result_callback = None


def init_ocr():
    """
    【兼容接口】预加载OCR模型（全局单例模式）
    
    供外部文件调用，避免首次识别时的模型加载延迟。
    内部自动复用已加载的引擎实例。
    
    Returns:
        OCREngineAccelerated: 已初始化的OCR引擎实例
        
    使用示例:
        from ocr_workflow_accelerated import init_ocr
        engine = init_ocr()  # 首次调用会加载模型(约0.2s)，后续调用直接返回已有实例
        result = engine.recognize("photo.jpg")
    """
    global _global_engine
    if _global_engine is None or _global_engine.det_session is None:
        print("[OCR Interface] Initializing OCR engine...")
        _global_engine = OCREngineAccelerated()
        _global_engine.init_model()
        print(f"[OCR Interface] Engine ready! (load time: {_global_engine._load_time:.3f}s)")
    return _global_engine


def recognize_image(image_path_or_array):
    """
    【标准对外接口】识别单张图片并返回文本结果
    
    供其他py文件调用的主入口。内部自动完成模型初始化、图片预处理、
    文字检测+分类+识别、版面排序、关键词过滤等全部流程。
    
    Args:
        image_path_or_array: 图片文件路径(str) 或 numpy数组(BGR格式, HWC)
        
    Returns:
        str: 识别出的完整文本（按阅读顺序排列，每行一个文字块）
             识别失败或无文字时返回空字符串 ""
             
    使用示例:
        from ocr_workflow_accelerated import recognize_image
        
        # 方式1: 传入图片路径
        text = recognize_image("photos/photo_1.jpg")
        print(text)
        
        # 方式2: 传入numpy数组 (cv2读取后的img)
        import cv2
        img = cv2.imread("photos/photo_1.jpg")
        text = recognize_image(img)
        print(text)
    """
    engine = init_ocr()
    
    # 云端优先模式 + 本地降级
    if isinstance(image_path_or_array, str) and MODE == 1:
        result = recognize_with_fallback(engine, image_path_or_array)
    else:
        result = engine.recognize(image_path_or_array)
        if result:
            result['source'] = 'local'
    
    if result is None or not result.get('texts'):
        return ""
    
    # 拼接所有文本为完整字符串（每行一个文字块）
    full_text = "\n".join(result['texts'])
    
    # 调用外部回调（如果已设置）
    global _ocr_result_callback
    if _ocr_result_callback:
        try:
            _ocr_result_callback(result, full_text)
        except Exception as e:
            print(f"[OCR Interface] Callback error: {e}")
    
    return full_text


def recognize_image_full(image_path_or_array):
    """
    【完整结果接口】识别单张图片并返回完整结构化结果
    
    与 recognize_image 的区别：返回包含坐标、置信度、耗时等全部信息的字典，
    适合需要后处理（如关键字提取、区域定位）的场景。
    
    Args:
        image_path_or_array: 图片文件路径(str) 或 numpy数组(BGR格式)
        
    Returns:
        dict: 完整识别结果 {
            'texts': [str, ...],       # 识别文本列表（按阅读顺序）
            'scores': [float, ...],     # 各文本置信度
            'boxes': [ndarray, ...],    # 各文本框坐标 (4, 2)
            'count': int,               # 有效文本数量
            'avg_score': float,         # 平均置信度
            'elapsed': float,           # 总耗时(秒)
            'elapsed_det': float,       # 检测耗时
            'elapsed_cls': float,       # 分类耗时
            'elapsed_rec': float,       # 识别耗时
        }
        或 None（图片无效时）
        
    使用示例:
        from ocr_workflow_accelerated import recognize_image_full
        
        result = recognize_image_full("photo.jpg")
        if result:
            print(f"识别到 {result['count']} 段文字, 平均置信度 {result['avg_score']:.3f}")
            for text, score in zip(result['texts'], result['scores']):
                print(f"  [{score:.2f}] {text}")
    """
    engine = init_ocr()
    
    # 云端优先模式 + 本地降级
    if isinstance(image_path_or_array, str) and OCR_ENGINE == 1:
        result = recognize_with_fallback(engine, image_path_or_array)
    else:
        result = engine.recognize(image_path_or_array)
        if result:
            result['source'] = 'local'
    
    # 触发回调
    global _ocr_result_callback
    if _ocr_result_callback and result:
        try:
            full_text = "\n".join(result.get('texts', []))
            _ocr_result_callback(result, full_text)
        except Exception as e:
            print(f"[OCR Interface] Callback error: {e}")
    
    return result


def batch_recognize(image_paths, output_txt=None):
    """
    【批量识别接口】批量识别多张图片
    
    Args:
        image_paths: 图片路径列表 [str, ...]
        output_txt: 可选，结果保存路径。提供时自动写入txt报告
        
    Returns:
        list[dict]: 每张图片的识别结果列表 [
            {'path': str, 'text': str, 'count': int, 'avg_score': float,
             'elapsed': float, 'success': bool}, ...
        ]
        
    使用示例:
        from ocr_workflow_accelerated import batch_recognize
        
        results = batch_recognize([
            'photos/photo_1.jpg',
            'photos/photo_10.jpg',
            'photos/photo_12.jpg',
        ], output_txt='ocr_batch_result.txt')
        
        for r in results:
            print(f"{r['path']}: {r['count']} texts, {r['elapsed']:.2f}s")
    """
    engine = init_ocr()
    all_results = []
    
    for path in image_paths:
        r = {
            'path': path,
            'text': '',
            'count': 0,
            'avg_score': 0.0,
            'elapsed': 0.0,
            'success': False,
        }
        
        if not os.path.exists(path):
            all_results.append(r)
            continue
            
        result = engine.recognize(path)
        if result and result.get('texts'):
            r['text'] = "\n".join(result['texts'])
            r['count'] = result['count']
            r['avg_score'] = result['avg_score']
            r['elapsed'] = result['elapsed']
            r['success'] = True
        all_results.append(r)
    
    # 写入txt报告
    if output_txt:
        run_test_and_save_report(engine, image_paths, output_txt)
    
    return all_results


def set_ocr_callback(callback_func):
    """
    设置OCR识别结果回调函数
    
    每次调用 recognize_image / recognize_image_full 识别完成后，
    会自动调用此回调函数传递结果。
    
    Args:
        callback_func: 回调函数，签名为 func(result_dict: dict, full_text: str) -> None
            - result_dict: 完整结构化结果（同 recognize_image_full 返回值）
            - full_text: 纯文本拼接结果（同 recognize_image 返回值）
            
    使用示例:
        from ocr_workflow_accelerated import set_ocr_callback, recognize_image
        
        def on_ocr_result(result, text):
            print(f"识别到 {result['count']} 段文字")
            # ... 发送给LLM / 显示到UI / 保存数据库 ...
            
        set_ocr_callback(on_ocr_result)
        text = recognize_image("photo.jpg")  # 识别完成后自动触发 on_ocr_result
    """
    global _ocr_result_callback
    _ocr_result_callback = callback_func
    tag = callback_func.__name__ if callback_func else None
    print(f"[OCR Interface] Callback set: {tag}")


def get_ocr_status():
    """
    查询接口：获取当前OCR引擎状态
    
    Returns:
        dict: {
            'initialized': bool,      # 模型是否已加载
            'load_time': float,       # 模型加载耗时(秒)，未加载时为0
            'det_model': str,         # 检测模型文件名
            'rec_model': str,         # 识别模型文件名
            'cls_model': str,         # 分类模型文件名
            'dict_size': int,         # 字典大小
            'acceleration': {         # 加速配置
                'intra_threads': int,
                'inter_threads': int,
                'rec_batch': int,
            },
        }
    """
    global _global_engine
    info = {
        'initialized': False,
        'load_time': 0.0,
        'det_model': DET_ONNX,
        'rec_model': REC_ONNX,
        'cls_model': CLS_ONNX,
        'dict_size': _PRELOAD_DICT_SIZE,
        'acceleration': {
            'intra_threads': ORT_INTRA_THREADS,
            'inter_threads': ORT_INTER_THREADS,
            'rec_batch': REC_BATCH_NUM,
        },
    }
    
    if _global_engine and _global_engine.det_session is not None:
        info['initialized'] = True
        info['load_time'] = _global_engine._load_time
    
    return info


# ============================================================
#  主入口
# ============================================================

def main():
    # 云端模式不需要加载本地模型，只在降级时再加载
    engine = None

    def _ensure_local_engine():
        """懒加载本地OCR引擎（仅在需要降级时才加载）"""
        nonlocal engine
        if engine is None:
            print("\n[本地OCR] 正在加载模型...")
            engine = OCREngineAccelerated()
            engine.init_model()
        return engine

    try:
        if MODE == 2:
            # 摄像头模式必须用本地OCR（实时性要求高）
            print("\n[模式2] 摄像头拍照识别 (本地OCR)")
            _ensure_local_engine()
            camera_mode(engine)
        elif MODE == 1:
            if not IMAGE_PATHS or all(p.strip().startswith('#') for p in IMAGE_PATHS):
                print("!!! 配置区 IMAGE_PATHS 为空，请添加图片路径 !!!")
                return

            paths = [p for p in IMAGE_PATHS if p and not p.strip().startswith('#')]
            mode_label = "云端优先(失败降级本地)" if MODE == 1 else "纯本地"
            print(f"\n[模式1] 图片识别 ({len(paths)} 张) | 引擎: {mode_label}")
            total_start = time.perf_counter()
            for path in paths:
                if not os.path.exists(path):
                    print(f"[跳过] 文件不存在: {path}")
                    continue
                print(f"{'='*60}")
                print(f"  图片: {path}")
                print(f"{'='*60}")

                # 统一识别入口
                if MODE == 1:
                    result = recognize_with_fallback(_ensure_local_engine(), path)
                else:
                    result = _ensure_local_engine().recognize(path)
                    if result:
                        result['source'] = 'local'

                if result:
                    print_result(result, path)
                else:
                    print("  [未识别到文字]\n")
            total_time = time.perf_counter() - total_start
            print(f"{'='*60}")
            print(f"[完成] 总耗时 {total_time:.3f}s")
            print(f"{'='*60}")
        else:
            print(f"!!! MODE 只能填 1 或 2，你填了 {MODE} !!!")
    finally:
        pass


if __name__ == '__main__':
    main()
