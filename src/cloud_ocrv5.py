import base64
import os
import time
import requests

API_URL = "https://z3c479c7k918obg7.aistudio-app.com/ocr"
TOKEN = "71bf88a8876217646a15669adb25fa0396d3e709"

TEST_IMAGES = [f"photos/photo_{i}.jpg" for i in range(4, 13)]
output_dir = "cloud_ocr_output/ocr_api"
os.makedirs(output_dir, exist_ok=True)

results_summary = []

for img_path in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"[SKIP] not found: {img_path}")
        continue

    print("=" * 60)
    print(f"  [OCR-API] Image: {img_path}")
    print("=" * 60)
    
    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")
    t_encode = time.perf_counter() - t0

    headers = {
        "Authorization": f"token {TOKEN}",
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
    response = requests.post(API_URL, json=payload, headers=headers)
    t_net = time.perf_counter() - t1
    print(f"  Encode: {t_encode:.3f}s | Request: {t_net:.3f}s | Status: {response.status_code}")

    if response.status_code != 200:
        print(f"  ERROR: {response.text[:300]}")
        print()
        continue

    result = response.json()["result"]
    ocr_results = result.get("ocrResults", [])
    input_filename = os.path.splitext(os.path.basename(img_path))[0]

    all_text_lines = []
    t2 = time.perf_counter()
    for i, res in enumerate(ocr_results):
        raw = res.get("prunedResult", "")
        if isinstance(raw, dict):
            # rec_texts 是识别出的文字列表
            rec_texts = raw.get("rec_texts", [])
            all_text_lines.extend(rec_texts)
        elif isinstance(raw, str):
            all_text_lines.append(raw)
        image_url = res.get("ocrImage", "")
        if image_url:
            try:
                img_resp = requests.get(image_url, timeout=10)
                if img_resp.status_code == 200:
                    fname = os.path.join(output_dir, f"{input_filename}_{i}.jpg")
                    with open(fname, "wb") as f:
                        f.write(img_resp.content)
            except Exception:
                pass
    t_save = time.perf_counter() - t2

    full_text = "\n".join(all_text_lines)
    total = time.perf_counter() - t0
    
    txt_path = os.path.join(output_dir, f"{input_filename}_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  Save: {t_save:.3f}s | Lines: {len(all_text_lines)} | Chars: {len(full_text)}")
    print(f"  Total: {total:.3f}s")
    print(f"  --- 识别结果预览(前500字) ---")
    print(full_text[:500])
    print()

    results_summary.append({
        'image': img_path,
        'total': total,
        'lines': len(all_text_lines),
        'chars': len(full_text),
        'preview': full_text[:500],
    })

print("=" * 60)
print("OCR-API Done")
print("=" * 60)
