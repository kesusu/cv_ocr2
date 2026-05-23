"""
PaddleOCR PP-OCRv4 文字识别测试
依赖: pip install paddleocr paddlepaddle
"""

from paddleocr import PaddleOCR

# -------------------- 配置 --------------------
test_images = [
    "photos/photo_12.jpg",
    #"photos/photo_2.jpg",
]

# -------------------- 初始化模型 --------------------
print("正在初始化 PaddleOCR PP-OCRv4 模型...")
print("(首次运行会下载模型，请耐心等待...)\n")

# 使用 PP-OCRv4 模型
ocr = PaddleOCR(
    use_angle_cls=True,      # 启用方向分类
    lang='ch',                # 中文
    use_gpu=False,            # 使用 CPU
    show_log=False            # 关闭日志
)

print("模型初始化完成！")
print("=" * 60)

# -------------------- 测试 --------------------
for img_path in test_images:
    print(f"\n测试图片: {img_path}")
    print("-" * 40)
    
    try:
        # 执行 OCR
        result = ocr.ocr(img_path)
        
        if not result or not result[0]:
            print("  [未检测到文字]")
            continue
        
        # 提取文字
        texts = []
        for line in result[0]:
            text = line[1][0]  # 文字内容
            confidence = line[1][1]  # 置信度
            texts.append((text, confidence))
            print(f"  {text}")
        
        print(f"\n  共识别 {len(texts)} 个文本区域")
        
    except Exception as e:
        print(f"  [错误] {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成！")
