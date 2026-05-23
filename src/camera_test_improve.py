# -*- coding: utf-8 -*-
"""
USB Camera - IMX577 (v6: 极简版)

  复盘结论 (2026-04-29):
    之前v3/v4/v5都在折腾硬件调参和激进软件处理, 全走错了。
    
    事实:
      b=0 + AE=ON(默认) → mean≈168, 画面虽然亮但文字可读
      设任何 brightness/exposure/AE 参数 → 反而更差
      IMX577+MSMF 驱动的硬件参数基本不可控
    
    策略: 
      ★ 完全不动硬件参数! 只设分辨率!
      ★ 软件只做轻量 gamma 压暗(可选)
      ★ 让摄像头自己管曝光, OCR引擎自己管识别
  
  功能:
    - 多后端自动回退
    - 空格=拍照(保存原图), Q/ESC=退出
    - P键=切换是否显示预处理效果(仅预览用)
"""

import cv2
import os
import time
import re
import numpy as np

SAVE_DIR = 'photos'

# ============================================================
#  轻量预处理 (仅用于预览对比/拍照时可选)
# ============================================================

def light_gamma(img, gamma=1.35):
    """
    极轻量 gamma 校正: 把偏亮的画面稍微压暗
    
    适用场景: mean > 150 时, gamma>1 可以把整体压下来
              文字从"灰白"变"清晰"
    
    不做 auto levels / 不做激进 CLAHE — 那些在动态范围窄时反而有害
    """
    if gamma == 1.0:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype='uint8')
    return cv2.LUT(img, table)


def get_stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return g.mean(), g.std()


# ============================================================
#  摄像头初始化 (★ 只设分辨率, 不动其他参数!)
# ============================================================

def find_best_camera():
    """多后端回退"""
    print('Opening camera...')
    
    backends = [
        (cv2.CAP_MSMF, 'MSMF'),
        (cv2.CAP_DSHOW, 'DSHOW'),
        (None, 'default'),
    ]
    
    for idx in range(3):
        for backend_id, name in backends:
            try:
                cap = cv2.VideoCapture(idx) if backend_id is None else \
                      cv2.VideoCapture(idx, backend_id)
                if not cap.isOpened():
                    continue
                    
                # ★ 只设这两个参数, 其他全不动!
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                # 预热
                ok_frames = 0
                for _ in range(20):
                    r, f = cap.read()
                    if r and f is not None and f.size > 0:
                        ok_frames += 1
                
                if ok_frames >= 10:
                    return idx, name, cap
                cap.release()
            except Exception:
                pass
    
    return 0, None, None


def _next_photo_id(d):
    if not os.path.exists(d): return 0
    mx = 0
    for f in os.listdir(d):
        m = re.match(r'photo_(\d+)\.', f)
        if m: mx = max(mx, int(m.group(1)))
    return mx + 1


# ============================================================
#  主程序
# ============================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 50)
    print("  USB Camera v6 (Minimal)")
    print("  Hardware params: DEFAULT (don't touch!)")
    print("=" * 50)

    idx, backend, cap = find_best_camera()
    if cap is None:
        input("\nCamera failed. Enter to exit.")
        return

    # 不调用 setup_camera! 保持默认状态

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 读一帧看初始状态
    for _ in range(10): cap.read()
    ret, sample = cap.read()
    init_mean, init_std = get_stats(sample) if ret else (0, 0)

    win = 'Camera'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    show_processed = False   # 默认显示原始画面
    photo_count = _next_photo_id(SAVE_DIR)
    fps_list, last_t, fps_v = [], time.time(), 0

    print(f"\n[OK] {w}x{h} | mean={init_mean:.0f} std={init_std:.1f}")
    print("-" * 45)
    print("  SPACE -> 拍照(保存原图)")
    print("  P     -> 切换 预览模式(gamma压暗)")
    print("  Q/ESC -> 退出")
    print("-" * 45)

    while True:
        try:
            ret, frame = cap.read()
        except cv2.error:
            if cv2.waitKey(50) & 0xFF in (ord('q'), 27): break
            continue

        if not ret or frame is None:
            if cv2.waitKey(30) & 0xFF in (ord('q'), 27): break
            continue

        now = time.time()
        fps_list.append(now)
        fps_list[:] = [t for t in fps_list if now - t < 1.0]
        if now - last_t >= 1.0:
            fps_v = len(fps_list); last_t = now

        # 显示选择
        display = light_gamma(frame, gamma=1.35) if show_processed else frame
        m, s = get_stats(display)

        mode = "[GAMMA]" if show_processed else "[RAW  ]"
        lines = [
            f"{w}x{h} | {fps_v}fps {mode}",
            f"mean={m:.0f} std={s:.1f}",
            f"Photos: {photo_count}",
            "SPACE=shot P=preview Q=quit"
        ]
        y = 28
        for line in lines:
            cv2.putText(display, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 2)
            y += 26

        cv2.imshow(win, display)
        key = cv2.waitKey(25) & 0xFF

        if key == ord(' '):
            photo_count += 1
            path = os.path.join(SAVE_DIR, f'photo_{photo_count}.jpg')
            # ★ 保存原始帧, 不做预处理(让OCR引擎决定怎么处理)
            ok, buf = cv2.imencode('.jpg', frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                with open(path, 'wb') as f:
                    f.write(buf)
                om, os_ = get_stats(frame)
                print(f"  [SAVED] photo_{photo_count}.jpg "
                      f"({os.path.getsize(path)/1024:.0f}KB) "
                      f"mean={om:.0f} std={os_:.1f}")
            
            flash = np.ones_like(frame) * 255
            cv2.imshow(win, flash)
            cv2.waitKey(80)

        elif key == ord('p'):
            show_processed = not show_processed

        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {photo_count} photos.")


if __name__ == '__main__':
    main()
