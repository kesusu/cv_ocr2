import cv2
import os
import time


# ============================================================
#  USB 外接摄像头 (芯片: IMX577)
#  - 自动选择最高分辨率的摄像头
#  - 分辨率: 1920x1080 @ 30fps
#  - MJPG格式 (摄像头原生支持)
# ============================================================

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photos')


def find_best_camera():
    """自动扫描所有摄像头，选择支持最高分辨率的那个"""
    print("Scanning cameras...")
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
                    mw, mh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    res = mw * mh
                    print(f"  [{i}] max: {mw}x{mh}")
                    if res > best_res:
                        best_res, best_idx = res, i
            cap.release()

    if best_idx is None:
        return None
    print(f"  -> Selected index {best_idx}\n")
    return best_idx


def setup_camera(cap):
    """
    设置摄像头参数 - 只设分辨率/格式/MJPG, 不动进光量参数
    (亮度/曝光/增益由摄像头AE自动管理, 避免过曝和拖影)
    """
    # --- 分辨率 & 帧率 ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # --- MJPG格式 (原生支持, 清晰度最好) ---
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # --- 以下参数不再手动设置 ---
    # BRIGHTNESS / EXPOSURE / GAIN: 由摄像头自动AE管理
    #   设定值会导致: ① 过曝(已验证) ② 曝光时间过长→拖影
    # SHARPNESS / CONTRAST / SATURATION: 保持出厂默认
    # AUTO_WB: 默认开启


def _sharpness_score(frame):
    """
    计算图像清晰度评分 (Laplacian 方差)
    值越大 = 边缘越锐利 = 越清晰
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _smart_shutter(cap, wait_ms=350, burst=5):
    """
    智能快门: 解决手持拍照拖影问题
    
    流程:
      1. 等待 wait_ms 毫秒 (让手停稳)
      2. 连续抓取 burst 帧
      3. 用 Laplacian 方差选最清晰的一帧
    
    参数:
        wait_ms:   按下快门后的等待时间(默认350ms)
        burst:     连续抓拍帧数(默认5帧, 约167ms@30fps)
    
    返回: 最清晰的 frame
    """
    # 阶段1: 等待稳定 (丢弃缓冲区中的旧帧)
    start = time.time()
    while (time.time() - start) * 1000 < wait_ms:
        cap.read()

    # 阶段2: 连续抓拍多帧, 选最清晰的
    best_frame = None
    best_score = -1

    for _ in range(burst):
        ret, frame = cap.read()
        if ret and frame is not None:
            score = _sharpness_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

    # 回退: 如果所有帧都失败, 读一帧返回
    if best_frame is None:
        ret, best_frame = cap.read()
        if not ret or best_frame is None:
            return None

    return best_frame


def _get_next_photo_id(save_dir):
    """扫描目录，返回下一个可用的编号（避免覆盖已有图片）"""
    if not os.path.exists(save_dir):
        return 0
    max_id = 0
    for f in os.listdir(save_dir):
        # 匹配 photo_N.jpg / photo_N.png 等格式
        import re
        m = re.match(r'photo_(\d+)\.', f)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 50)
    print("       USB Camera - IMX577")
    print("=" * 50)
    print("  Resolution: 1920x1080 @ 30fps")
    print("  Format: MJPG (native)")
    print("=" * 50 + "\n")

    CAMERA_INDEX = find_best_camera()
    if CAMERA_INDEX is None:
        print("ERROR: No camera found!")
        input("Press Enter to exit...")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        input("Press Enter to exit...")
        return

    setup_camera(cap)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    win = 'Camera'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    print(f"[OK] {w}x{h} @ 30fps | Format: {fourcc_str}")
    print("  Smart Shutter: ON (anti-shake, 5-frame burst)")
    print("\n  SPACE -> Take photo (auto anti-shake)")
    print("  Q     -> Quit\n")

    photo_count = _get_next_photo_id(SAVE_DIR)   # 自动接续已有编号，不覆盖
    fps_list, t_last, fps_show = [], time.time(), 0

    has_psutil = True
    try:
        import psutil
    except:
        has_psutil = False

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
            f"{w}x{h}  |  {fps_show}fps  |  {fourcc_str}",
            f"CPU:{cpu_v:.0f}%  RAM:{mem_v:.0f}%  App:{proc_v:.0f}MB",
            f"Photos: {photo_count}",
            "SPACE:shot  Q:quit"
        ]

        y = 28
        for line in info:
            cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 120), 2)
            y += 28

        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord(' '):
            # 智能快门: 等待稳定 + 多帧选最清晰
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
            else:
                print(" FAILED - no frame")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {photo_count} photos saved.")


if __name__ == '__main__':
    main()
