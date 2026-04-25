# -*- coding: utf-8 -*-
"""
Rec 加速基准测试 — PC 端独立运行脚本 (纯 ORT 直接测试)

验证 PROJECT_SUMMARY.md 第八章中的优化方案:
  1. ORT SessionOptions 调优 (线程/执行模式/内存)
  2. 不同批处理大小
  3. 不同输入宽度 (320/256/192) — 精度/速度权衡

用法: python bench_rec_pc.py
结果输出到: bench_result.txt

注意: 此脚本直接使用 onnxruntime 测试 Rec 模型，
     不依赖 RapidOCR，避免 YAML 配置兼容性问题。
"""
import sys
import os
import time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import onnxruntime as ort

# ── 配置 ──
MODEL_DIR = os.path.join(BASE_DIR, 'pp-ocrv4_rapid_onnx')
REC_MODEL = os.path.join(MODEL_DIR, 'ch_PP-OCRv4_rec_mobile.onnx')
TEST_IMAGE = os.path.join(BASE_DIR, 'photos', 'photo_1.jpg')
RESULT_FILE = os.path.join(BASE_DIR, 'bench_result.txt')

WARMUP = 2
REPEATS = 5


def make_dummy_input(batch, height=48, width=320):
    """生成 Rec 模型的 dummy 输入 [B, C, H, W]"""
    return np.random.randn(batch, 3, height, width).astype(np.float32)


def bench_config(session, input_shape, label, batch):
    """对给定 session 运行 benchmark"""
    dummy = make_dummy_input(batch, input_shape[1], input_shape[2])
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(WARMUP):
        session.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    per_sample = avg_ms / batch
    return avg_ms, p95_ms, per_sample


def create_session(threads=-1, sequential=False, mem_reuse=False):
    """创建带指定优化选项的 ORT Session"""
    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
    if sequential:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if mem_reuse:
        so.enable_mem_reuse = True
    so.enable_mem_pattern = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(REC_MODEL, so)


def run_benchmark():
    if not os.path.exists(REC_MODEL):
        print(f"[ERROR] Rec model not found: {REC_MODEL}")
        return

    lines = []
    def log(msg=''):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("  PP-OCRv4 Rec Acceleration Benchmark (Direct ORT)")
    log(f"  Model: {os.path.basename(REC_MODEL)}")
    log(f"  Warmup={WARMup}, Repeats={REPEATS}")
    log("=" * 72)

    # ── 定义测试矩阵 ──
    tests = [
        # (label,                threads, seq, memreuse, width, batch, desc)
        ("1.Default",           -1,     False, False,  320, 6,  "RapidOCR default equivalent"),
        ("2.ORT Optimized",      4,      True,  True,   320, 6,  "sequential+mem_reuse (ocr_board.py config)"),
        ("3.Thread=1",          1,      False, False,  320, 6,  "single-threaded baseline"),
        ("4.Thread=4",          4,      False, False,  320, 6,  "multi-thread only"),
        ("5.Batch=12",          4,      True,  True,   320, 12, "larger batch"),
        ("6.Width=256",         4,      True,  True,   256, 6,  "-20% compute"),
        ("7.Width=192",         4,      True,  True,   192, 6,  "-40% compute"),
        ("8.W256+Batch12",      4,      True,  True,   256, 12, "combined best"),
    ]

    results = []

    for label, thr, seq, mr, w, b, desc in tests:
        log(f"\n--- [{label}] {desc} (w={w}, batch={b}, thr={thr}, seq={seq}, mr={mr}) ---")
        try:
            sess = create_session(threads=thr, sequential=seq, mem_reuse=mr)
            avg, p95, ps = bench_config(sess, (b, 3, 48, w), label, b)
            results.append({
                'label': label, 'width': w, 'batch': b,
                'thr': thr, 'seq': seq, 'mr': mr,
                'avg_ms': avg, 'p95_ms': p95, 'per_sample_ms': ps,
            })
            log(f"  avg={avg:.1f}ms  p95={p95:.1f}ms  per-sample={ps:.2f}ms")
            del sess
        except Exception as e:
            log(f"  [ERROR] {e}")

    # ── Summary table ──
    log("\n" + "=" * 82)
    log("  BENCHMARK RESULTS")
    log("=" * 82)
    header = f"{'Config':<22s} {'W':>3s} {'Bat':>4s} {'Thr':>4s} {'Seq':>3s} {'MR':>3s} {'Avg(ms)':>8s} {'P95(ms)':>8s} {'/sample':>8s} {'Speed':>6s}"
    log(header)
    log("-" * 82)

    baseline_avg = None
    for r in results:
        if r['label'].startswith('1.Default'):
            baseline_avg = r['avg_ms']
            speed = "1.00x"
        elif baseline_avg and baseline_avg > 0:
            speed = f"{baseline_avg/r['avg_ms']:.2f}x"
        else:
            speed = "N/A"

        log(f"{r['label']:<22s} {r['width']:>3d} {r['batch']:>4d} "
            f"{r['thr']:>4d} {'Y' if r['seq'] else 'N':>3s} "
            f"{'Y' if r['mr'] else 'N':>3s} "
            f"{r['avg_ms']:>8.1f} {r['p95_ms']:>8.1f} "
            f"{r['per_sample_ms']:>8.2f} {speed:>6s}")

    log("-" * 82)

    # ── Analysis ──
    log("\n  [ANALYSIS]")

    if len(results) >= 2:
        best = min(results, key=lambda x: x['avg_ms'])
        log(f"     Fastest overall: [{best['label']}] ({best['avg_ms']:.1f}ms total, "
            f"{best['per_sample_ms']:.2f}ms/sample)")

        if baseline_avg:
            imp = (baseline_avg - best['avg_ms']) / baseline_avg * 100
            log(f"     vs Default improvement: {imp:+.1f}%")

        # Thread comparison
        t_def = next((r for r in results if 'Default' in r['label']), None)
        t1 = next((r for r in results if 'Thread=1' in r['label']), None)
        t4 = next((r for r in results if 'Thread=4' in r['label']), None)
        opt = next((r for r in results if 'ORT Optimized' in r['label']), None)

        if t1 and t_def:
            log(f"\n     --- Thread Scaling (width=320, batch=6) ---")
            log(f"       Thread=-1 (auto): {t_def['avg_ms']:.1f}ms")
            log(f"       Thread=1:         {t1['avg_ms']:.1f}ms ({t_def['avg_ms']/t1['avg_ms']:.2f}x slower)")
            if t4:
                log(f"       Thread=4:         {t4['avg_ms']:.1f}ms")
            if opt:
                log(f"       Thread=4+Opt:     {opt['avg_ms']:.1f}ms ({t_def['avg_ms']/opt['avg_ms']:.2f}x vs default)")

        # Width comparison (both with optimized settings)
        w320_opt = next((r for r in results if r['width'] == 320 and r['batch'] == 6 and r.get('seq')), None)
        w256_opt = next((r for r in results if r['width'] == 256 and r['batch'] == 6 and r.get('seq')), None)
        w192_opt = next((r for r in results if r['width'] == 192 and r['batch'] == 6 and r.get('seq')), None)

        if w320_opt and w256_opt:
            w_speedup = w320_opt['avg_ms'] / w256_opt['avg_ms']
            log(f"\n     --- Width Reduction (optimized, batch=6) ---")
            log(f"       Width=320: {w320_opt['avg_ms']:.1f}ms ({w320_opt['per_sample_ms']:.2f}ms/sample)")
            log(f"       Width=256: {w256_opt['avg_ms']:.1f}ms ({w256_opt['per_sample_ms']:.2f}ms/sample) [{w_speedup:.2f}x]")
            if w192_opt:
                w192_speedup = w320_opt['avg_ms'] / w192_opt['avg_ms']
                log(f"       Width=192: {w192_opt['avg_ms']:.1f}ms ({w192_opt['per_sample_ms']:.2f}ms/sample) [{w192_speedup:.2f}x]")

        # Batch comparison
        b6_opt = next((r for r in results if r['width'] == 320 and r['batch'] == 6 and r.get('seq')), None)
        b12_opt = next((r for r in results if r['width'] == 320 and r['batch'] == 12 and r.get('seq')), None)
        if b6_opt and b12_opt:
            b6_ps = b6_opt['per_sample_ms']
            b12_ps = b12_opt['per_sample_ms']
            log(f"\n     --- Batch Size (width=320, optimized) ---")
            log(f"       batch=6:  {b6_opt['avg_ms']:.1f}ms total, {b6_ps:.2f}ms/sample")
            log(f"       batch=12: {b12_opt['avg_ms']:.1f}ms total, {b12_ps:.2f}ms/sample ({b6_ps/b12_ps:.2f}x per-sample)")

    log("\n" + "=" * 82)
    log("  RECOMMENDED CONFIG for ocr_board.py:")
    log(f"     ORT_THREADS = 4       # or match CPU core count")
    log(f"     # In _init_pc_fallback():")
    log(f"     #   execution_mode = SEQUENTIAL")
    log(f"     #   enable_mem_reuse = True")
    log(f"     REC_BATCH_NUM = ???   # see width comparison above")
    log(f"     REC_IMAGE_WIDTH = ??? # 320=default, 256=~20%% faster, 192=~40%% faster")
    log("=" * 82)

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"\n  Results saved to: {RESULT_FILE}")


if __name__ == '__main__':
    run_benchmark()
