"""
自定义校准集 INT8 量化脚本 — Rec 模型专用
================================================
用途: 用项目实际图片(药品说明书)做校准集, 解决官方INT8模型校准不匹配问题

两种模式:
  Mode A: ort QDQ 静态量化 (PC直接跑, 不需DLC)
           结果: 精度好, 但 CPU 推理时可能无加速(QDQ→反量化FP32计算)
           
  Mode B: 生成校准数据集 + 指引到 Docker 做 SNPE 量化
           结果: 可生成真正在 DSP/NPU 加速的 DLC

用法:
  python quantize_rec_int8.py              # 默认 Mode A: 直接量化
  python quantize_rec_int8.py --mode B      # Mode B: 只生成校准集
  
依赖:
  pip install onnx onnxruntime numpy Pillow
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np
from PIL import Image
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ============================================================
# 配置区 — 和 ocr_board.py / ocr_workflow_onnx.py 保持一致
# ============================================================
PROJECT_DIR = Path(__file__).parent.resolve()
ONNX_DIR = PROJECT_DIR / "pp-ocrv4_rapid_onnx"
REC_ONNX_FP32 = ONNX_DIR / "ch_PP-OCRv4_rec_mobile.onnx"
REC_ONNX_INT8_OUT = ONNX_DIR / "ch_PP-OCRv4_rec_mobile_custom_int8.onnx"

# Rec 参数 (必须和推理代码完全一致)
REC_IMAGE_HEIGHT = 48
REC_IMAGE_WIDTH = 320          # ← 后续可试 256
REC_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
REC_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# 校准图片源
CALIBRATION_DIRS = [
    PROJECT_DIR / "photos",                    # 项目自带测试图片
    PROJECT_DIR / "板端数据",                   # 如果有板端采集的图片
]
CALIBRATION_IMAGES_GLOB = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
CALIBRATION_SET_SIZE = 100                     # 目标校准样本数 (越多越好, 推荐50~200)


def find_calibration_images():
    """收集所有可用于校准的图片"""
    images = []
    for d in CALIBRATION_DIRS:
        if not d.is_dir():
            log.warning(f"校准图片目录不存在: {d}")
            continue
        for pattern in CALIBRATION_IMAGES_GLOB:
            found = list(d.glob(pattern))
            images.extend(found)
            log.info(f"从 {d/pattern} 找到 {len(found)} 张")
    
    # 去重
    images = sorted(set(images))
    log.info(f"共找到 {len(images)} 张校准候选图片")
    
    if len(images) == 0:
        raise FileNotFoundError(
            "没有找到任何校准图片! 请在 photos/ 或 板端数据/ 放入项目实际要识别的图片\n"
            "建议放入 10~50 张不同的药品说明书照片"
        )
    return images


def preprocess_for_rec(image_path):
    """
    预处理一张图片为 Rec 输入格式
    必须和推理时的预处理完全一致!
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((REC_IMAGE_WIDTH, REC_IMAGE_HEIGHT), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = (img_arr - REC_MEAN) / REC_STD
    # ONNX Rec 输入: [1, 3, H, W]
    img_arr = img_arr.transpose(2, 0, 1)[np.newaxis, ...]
    return img_arr


def generate_calibration_set(images, max_size=CALIBRATION_SET_SIZE):
    """
    生成校准集
    策略: 
      1. 对每张图做完整 OCR 流程(Det→裁剪→Rec预处理)
      2. 收集所有文本框 crop 后的 Rec 输入张量
      3. 这样校准集分布和真实推理分布一致
    """
    log.info("=" * 60)
    log.info("生成校准集...")
    log.info("=" * 60)
    
    # 尝试导入 Det 模型来裁剪文本框 (可选, 如果没有Det就用整图resize)
    calibration_data = []
    
    try:
        import onnxruntime as ort
        
        det_onnx = ONNX_DIR / "ch_PP-OCRv4_det_mobile.onnx"
        if not det_onnx.exists():
            raise FileNotFoundError("Det模型不存在, 回退到整图方案")
        
        log.info(f"加载 Det: {det_onnx}")
        det_sess = ort.InferenceSession(str(det_onnx), providers=['CPUExecutionProvider'])
        
        det_input_name = det_sess.get_inputs()[0].name
        det_output_names = [o.name for o in det_sess.get_outputs()]
        
        for img_path in images[:max_size]:  # 限制图片数量
            try:
                # Det 预处理
                img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = img.size
                max_side = max(orig_w, orig_h)
                ratio = 736.0 / max_side  # DET_LIMIT_SIDE_LEN
                new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
                img_resized = img.resize((new_w, new_h), Image.BILINEAR)
                img_arr = np.array(img_resized, dtype=np.float32)
                img_arr = (img_arr - [[127.5, 127.5, 127.5]]) / 127.5
                img_arr = img_arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1,3,H,W]
                
                # Det 推理
                outputs = det_sess.run(det_output_names, {det_input_name: img_arr})
                
                # 简化的后处理: 假设输出是 maps + masks (PP-OCRv4 mobile)
                if len(outputs) >= 2:
                    maps = outputs[0]  # [1, C, H', W']
                    
                    # 取置信度 > 阈值的区域作为文本框
                    conf_map = maps[0].max(axis=0)  # [H', W']
                    thresh = 0.3
                    ys, xs = np.where(conf_map > thresh)
                    
                    if len(xs) == 0:
                        # 没检测到框, 用整图作为校准样本
                        cal_input = preprocess_for_rec(img_path)
                        calibration_data.append(cal_input)
                        continue
                    
                    # 对每个检测到的区域, 裁剪并预处理为 Rec 输入
                    count = 0
                    for x, y in zip(xs[:20], ys[:20]):  # 每张图最多取20个框
                        # 映射回原图坐标
                        scale_x, scale_y = orig_w / new_w, orig_h / new_h
                        cx, cy = int(x * scale_x), int(y * scale_y)
                        
                        # 裁剪一个以该点为中心的区域 (模拟文本框crop)
                        half_w, half_h = min(80, orig_w // 4), min(20, orig_h // 4)
                        x1 = max(0, cx - half_w)
                        y1 = max(0, cy - half_h)
                        x2 = min(orig_w, cx + half_w)
                        y2 = min(orig_h, cy + half_h)
                        
                        crop = img.crop((x1, y1, x2, y2))
                        if crop.size[0] < 10 or crop.size[1] < 3:
                            continue
                        
                        # Resize 到 Rec 输入尺寸
                        crop_resized = crop.resize((REC_IMAGE_WIDTH, REC_IMAGE_HEIGHT), Image.BILINEAR)
                        crop_arr = np.array(crop_resized, dtype=np.float32) / 255.0
                        crop_arr = (crop_arr - REC_MEAN) / REC_STD
                        crop_arr = crop_arr.transpose(2, 0, 1)[np.newaxis, ...]
                        calibration_data.append(crop_arr)
                        count += 1
                    
                    if count == 0:
                        cal_input = preprocess_for_rec(img_path)
                        calibration_data.append(cal_input)
                else:
                    cal_input = preprocess_for_rec(img_path)
                    calibration_data.append(cal_input)
                    
            except Exception as e:
                log.debug(f"处理图片失败 {img_path.name}: {e}")
                # 回退: 用整图 resize
                try:
                    cal_input = preprocess_for_rec(img_path)
                    calibration_data.append(cal_input)
                except:
                    pass
        
        del det_sess
        
    except ImportError:
        log.warning("onnxruntime 未安装, 使用简化的整图 resize 方案生成校准集")
        for img_path in images[:max_size]:
            try:
                cal_input = preprocess_for_rec(img_path)
                calibration_data.append(cal_input)
            except Exception as e:
                log.debug(f"处理失败 {img_path.name}: {e}")
    
    except Exception as e:
        log.warning(f"Det 模型加载失败 ({e}), 使用简化的整图 resize 方案")
        for img_path in images[:max_size]:
            try:
                cal_input = preprocess_for_rec(img_path)
                calibration_data.append(cal_input)
            except:
                pass
    
    log.info(f"校准集生成完毕: {len(calibration_data)} 个样本")
    
    if len(calibration_data) < 10:
        log.warning(
            f"⚠️ 校准集只有 {len(calibration_data)} 个样本, 可能不够!\n"
            f"   建议: 放入更多项目实际图片到 photos/ 目录"
        )
    
    return calibration_data


class CalibrationDataset:
    """ORT quantize_static 需要的校准数据集类 (ORT 1.19+ 使用 get_next() 迭代器)"""
    def __init__(self, calibration_data, input_name):
        self.data = calibration_data
        self.input_name = input_name
        self.idx = 0
    
    def __len__(self):
        return len(self.data)
    
    def get_next(self):
        """ORT 1.19+ 校准数据读取器接口"""
        if self.idx >= len(self.data):
            return None
        result = {self.input_name: self.data[self.idx]}
        self.idx += 1
        return result
    
    def rewind(self):
        """重置迭代器 (ORT 可能多次遍历校准集)"""
        self.idx = 0


def run_ort_qdq_quantization(rec_onnx_path, output_path, calibration_data):
    """
    Mode A: ORT 静态量化 (QDQ)
    
    注意: 
      - PC (x86): QDQ 通常不会加速(CPU Execution Provider 反量化回FP32算)
      - ARM: 同样取决于是否支持 int8 execution provider
      - 但精度应该很好, 因为用的是自己的校准集
    """
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    
    log.info("=" * 60)
    log.info("Mode A: ORT 静态量化 (QDQ)")
    log.info("=" * 60)
    
    sess_input_name = None
    
    # 先获取输入名
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(rec_onnx_path), providers=['CPUExecutionProvider'])
        sess_input_name = sess.get_inputs()[0].name
        log.info(f"模型输入名: {sess_input_name}, shape: {sess.get_inputs()[0].shape}")
        del sess
    except Exception as e:
        log.error(f"无法读取模型输入信息: {e}")
        return False
    
    dataset = CalibrationDataset(calibration_data, sess_input_name)
    
    log.info(f"开始量化...")
    log.info(f"  FP32模型: {rec_onnx_path} ({rec_onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    log.info(f"  校准样本: {len(dataset)} 个")
    log.info(f"  输出路径: {output_path}")
    
    # 中文路径会导致 quantize_static 内部 shape inference 失败
    # 解决: 复制到临时英文路径做量化, 完成后复制回来
    import tempfile
    import shutil
    tmpdir = Path(tempfile.mkdtemp(prefix='ort_quant_', dir=Path(tempfile.gettempdir())))
    tmp_model = tmpdir / "model_fp32.onnx"
    tmp_output = tmpdir / "model_int8.onnx"
    
    try:
        shutil.copy2(str(rec_onnx_path), str(tmp_model))
        log.info(f"  (模型已复制到临时路径: {tmpdir})")
        
        quantize_static(
            model_input=str(tmp_model),
            model_output=str(tmp_output),
            calibration_data_reader=dataset,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            extra_options={
                'ActivationSymmetric': 'true',
                'WeightSymmetric': 'false',
            }
        )
        
        # 复制回目标路径
        shutil.copy2(str(tmp_output), str(output_path))
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    out_size = output_path.stat().st_size / 1024 / 1024
    inp_size = rec_onnx_path.stat().st_size / 1024 / 1024
    log.info(f"✅ 量化完成!")
    log.info(f"  FP32: {inp_size:.1f} MB → INT8(QDQ): {out_size:.1f} MB (压缩 {(1-out_size/inp_size)*100:.0f}%)")
    
    return True


def verify_quantized_model(int8_model_path, calibration_data=None):
    """验证量化后模型的精度和速度"""
    import onnxruntime as ort
    import time
    
    log.info("=" * 60)
    log.info("验证量化模型")
    log.info("=" * 60)
    
    fp32_model = str(REC_ONNX_FP32)
    int8_model = str(int8_model_path)
    
    # 准备测试数据
    if calibration_data and len(calibration_data) > 0:
        test_data = calibration_data[:min(20, len(calibration_data))]
        input_name = 'x'  # PP-OCRv4 Rec 默认输入名
    else:
        # 生成随机测试数据
        np.random.seed(42)
        test_data = [{input_name: np.random.randn(1, 3, 48, 320).astype(np.float32)}]
    
    results = {}
    
    for label, model_path in [("FP32", fp32_model), ("INT8-QDQ", int8_model)]:
        if not os.path.exists(model_path):
            log.warning(f"模型不存在, 跳过: {model_path}")
            continue
            
        try:
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            inp_name = sess.get_inputs()[0].name
            
            # Warmup
            sample = test_data[0][inp_name] if isinstance(test_data[0], dict) else test_data[0]
            _ = sess.run(None, {inp_name: sample})
            
            # Benchmark
            times = []
            outputs_fp32 = None
            for data in test_data:
                inp = data[inp_name] if isinstance(data, dict) else data
                
                t0 = time.perf_counter()
                outputs = sess.run(None, {inp_name: inp})
                t1 = time.perf_counter()
                
                times.append(t1 - t0)
                if outputs_fp32 is None:
                    outputs_fp32 = outputs
            
            avg_time = np.mean(times) * 1000
            results[label] = {
                'time_ms': avg_time,
                'outputs': outputs_fp32,
            }
            log.info(f"  {label}: 平均 {avg_time:.1f} ms/张 (batch=1)")
            
            del sess
            
        except Exception as e:
            log.error(f"  {label} 测试失败: {e}")
    
    # 对比精度
    if len(results) == 2:
        out_fp32 = results['FP32']['outputs'][0]
        out_int8 = results['INT8-QDQ']['outputs'][0]
        
        if out_fp32.shape == out_int8.shape:
            max_diff = np.abs(out_fp32 - out_int8).max()
            mean_diff = np.abs(out_fp32 - out_int8).mean()
            cos_sim = np.dot(out_fp32.flatten(), out_int8.flatten()) / (
                np.linalg.norm(out_fp32.flatten()) * np.linalg.norm(out_int8.flatten()) + 1e-10
            )
            
            log.info(f"  精度对比:")
            log.info(f"    最大绝对误差: {max_diff:.6f}")
            log.info(f"    平均绝对误差: {mean_diff:.6f}")
            log.info(f"    余弦相似度:   {cos_sim:.8f}")
            
            if max_diff < 0.1 and cos_sim > 0.99:
                log.info(f"  ✅ 精度损失极小, 量化质量良好")
            elif max_diff < 1.0 and cos_sim > 0.95:
                log.info(f"  ⚠️ 有一定精度损失, 但可能可用")
            else:
                log.warning(f"  ❌ 精度损失较大, 请检查校准集质量!")
            
            # 速度对比
            speedup = results['FP32']['time_ms'] / results['INT8-QDQ']['time_ms']
            log.info(f"  速度对比:")
            log.info(f"    FP32: {results['FP32']['time_ms']:.1f} ms")
            log.info(f"    INT8: {results['INT8-QDQ']['time_ms']:.1f} ms")
            if speedup > 1.05:
                log.info(f"    ✅ 加速 {speedup:.2f}x")
            elif speedup > 0.95:
                log.info(f"    ≈ 基本持平 (CPU可能不支持INT8指令加速QDQ)")
            else:
                log.warning(f"    ❌ 变慢 {1/speedup:.2f}x (QDQ开销 > INT8加速收益)")
    
    return results


def main():
    global REC_IMAGE_WIDTH
    parser = argparse.ArgumentParser(description='Rec 模型自定义校准集 INT8 量化')
    parser.add_argument('--mode', choices=['A', 'B'], default='A',
                        help='A: ORT QDQ静态量化(默认) | B: 仅生成校准集(给SNPE用)')
    parser.add_argument('--width', type=int, default=REC_IMAGE_WIDTH,
                        help=f'Rec 输入宽度 (默认{REC_IMAGE_WIDTH})')
    parser.add_argument('--verify', action='store_true',
                        help='量化完成后自动验证精度和速度')
    args = parser.parse_args()
    
    REC_IMAGE_WIDTH = args.width
    
    log.info("=" * 60)
    log.info("自定义校准集 INT8 量化 — Rec 模型")
    log.info("=" * 60)
    log.info(f"模式: {'A (ORT QDQ)' if args.mode == 'A' else 'B (仅生成校准集)'}")
    log.info(f"Rec 输入宽度: {REC_IMAGE_WIDTH}")
    
    # Step 1: 找图片
    images = find_calibration_images()
    
    # Step 2: 生成校准集
    calibration_data = generate_calibration_set(images)
    
    if args.mode == 'B':
        # Mode B: 保存校准集到文件, 供 Docker/SNPE 使用
        calib_dir = PROJECT_DIR / "calibration_data"
        calib_dir.mkdir(exist_ok=True)
        
        calib_file = calib_dir / f"rec_calib_set_w{REC_IMAGE_WIDTH}.npy"
        np.save(calib_file, np.concatenate(calibration_data, axis=0))
        log.info(f"✅ 校准集已保存: {calib_file}")
        log.info(f"   shape: {np.load(calib_file).shape}")
        log.info("")
        log.info("下一步: 在 Docker 中使用此校准集运行 SNPE 量化:")
        log.info(f"  docker exec my_work snpe-quantize-onnx \\")
        log.info(f"    --input_network {REC_ONNX_FP32.name} \\")
        log.info(f"    --calibration_data {calib_file.name} \\")
        log.info(f"    --output_network rec_int8_snpe.dlc")
        return
    
    # Mode A: ORT QDQ 量化
    if not REC_ONNX_FP32.exists():
        log.error(f"FP32 模型不存在: {REC_ONNX_FP32}")
        sys.exit(1)
    
    success = run_ort_qdq_quantization(REC_ONNX_FP32, REC_ONNX_INT8_OUT, calibration_data)
    
    if not success:
        log.error("量化失败!")
        sys.exit(1)
    
    # 验证
    if args.verify or True:  # 默认验证
        print()
        results = verify_quantized_model(REC_ONNX_INT8_OUT, calibration_data)
    
    log.info("")
    log.info("=" * 60)
    log.info("完成!")
    log.info("=" * 60)
    log.info(f"输出模型: {REC_ONNX_INT8_OUT}")
    log.info("")
    log.info("后续步骤:")
    log.info("  1. 将此模型复制到开发板")
    log.info("  2. 修改 ocr_board.py 中的 REC_ONNX 指向新模型")
    log.info("  3. 运行完整 OCR 测试对比精度和速度")
    log.info("")
    log.info("如果验证显示'变慢'且'基本持平', 说明 CPU 不支持 QDQ 加速:")
    log.info("  → 此时应考虑 Mode B (SNPE 量化) 或其他优化方向(P0/P1/P3)")


if __name__ == '__main__':
    main()
