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
    设置摄像头参数 - 优化清晰度
    """
    # --- 分辨率 & 帧率 ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # --- MJPG格式 (原生支持，清晰度最好) ---
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # --- 图像参数优化 ---
    # 降低增益减少噪点，提升锐度来弥补
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)   # 亮度适中
    cap.set(cv2.CAP_PROP_CONTRAST, 150)      # 提高对比度让细节更分明
    cap.set(cv2.CAP_PROP_SATURATION, 115)    # 饱和度稍低更自然

    # 锐度 (重要！提升清晰感)
    try:
        cap.set(cv2.CAP_PROP_SHARPNESS, 200)   # 提高锐度
    except:
        pass

    # 增益 (降低减少噪点)
    try:
        cap.set(cv2.CAP_PROP_GAIN, 100)        # 降低增益
    except:
        pass

    # 曝光 (让DSP自动处理)
    try:
        cap.set(cv2.CAP_PROP_EXPOSURE, -3)     # 稍暗一点，减少过曝
    except:
        pass

    # 白平衡 (让DSP自动处理)
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)        # 开启自动白平衡
    except:
        pass


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
    print("\n  SPACE -> Take photo")
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
            photo_count += 1
            path = os.path.join(SAVE_DIR, f'photo_{photo_count}.jpg')
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if ok:
                with open(path, 'wb') as f:
                    f.write(buf)
                sz = os.path.getsize(path) / 1024
                print(f"  [SAVED] photo_{photo_count}.jpg ({sz:.0f}KB)")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {photo_count} photos saved.")


if __name__ == '__main__':
    main()
