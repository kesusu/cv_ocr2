import base64
import os
import time
import requests

API_URL = "https://p975d56ew1m3u3h6.aistudio-app.com/layout-parsing"
TOKEN = "71bf88a8876217646a15669adb25fa0396d3e709"

TEST_IMAGES = ["photos/photo_1.jpg", "photos/photo_12.jpg"]
output_dir = "cloud_ocr_output/layout_v2"
os.makedirs(output_dir, exist_ok=True)

results_summary = []

for img_path in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"[SKIP] not found: {img_path}")
        continue

    print("=" * 60)
    print(f"  [Layout-V2] Image: {img_path}")
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
        "useChartRecognition": False,
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
    layout_results = result.get("layoutParsingResults", [])

    all_texts = []
    t2 = time.perf_counter()
    for i, res in enumerate(layout_results):
        md_text = res.get("markdown", {}).get("text", "")
        all_texts.append(md_text)
        input_filename = os.path.splitext(os.path.basename(img_path))[0]
        md_filename = os.path.join(output_dir, f"{input_filename}_{i}.md")
        with open(md_filename, "w", encoding="utf-8") as mf:
            mf.write(md_text)
        for img_name, img_url in res.get("outputImages", {}).items():
            try:
                img_resp = requests.get(img_url, timeout=10)
                if img_resp.status_code == 200:
                    fname = os.path.join(output_dir, f"{input_filename}_{img_name}_{i}.jpg")
                    with open(fname, "wb") as f:
                        f.write(img_resp.content)
            except Exception:
                pass
    t_save = time.perf_counter() - t2

    full_text = "\n".join(all_texts)
    total = time.perf_counter() - t0

    txt_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  Save: {t_save:.3f}s | Docs: {len(layout_results)} | Chars: {len(full_text)}")
    print(f"  Total: {total:.3f}s")
    print()

    results_summary.append({
        'image': img_path,
        'total': total,
        'docs': len(layout_results),
        'chars': len(full_text),
        'preview': full_text[:500],
    })

print("=" * 60)
print("Layout-V2 Done")
print("=" * 60)
