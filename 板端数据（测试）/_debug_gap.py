"""
诊断脚本: 对比 ocr_board.py 和 RapidOCR 的每一步输出
定位丢失的 7 个框 + 识别质量差异根因
"""
import sys
sys.path.insert(0, '/home/fibo/cv')
import cv2
import numpy as np
from rapidocr import RapidOCR

# ============================================================
# Part 1: 用 RapidOCR 获取参考结果
# ============================================================
print("=" * 60)
print("Part 1: Running RapidOCR (reference)")
print("=" * 60)

engine = RapidOCR()
output = engine('photos/photo_1.jpg')

rapid_boxes = []   # (box_array, text, score)
if output.boxes is not None:
    for i in range(len(output.boxes)):
        rapid_boxes.append((output.boxes[i], output.txts[i], float(output.scores[i])))

print(f"RapidOCR total: {len(rapid_boxes)} results")
print()

# ============================================================
# Part 2: 用 ocr_board 获取对比结果
# ============================================================
print("=" * 60)
print("Part 2: Running ocr_board.py")
print("=" * 60)

from ocr_board import OCRBoardEngine

board_engine = OCRBoardEngine()
board_engine.init_model()

bresult = board_engine.recognize('photos/photo_1.jpg')

board_texts = bresult['texts']
board_scores = bresult['scores']
board_boxes = bresult['boxes']

print(f"\nocr_board total: {len(board_texts)} results (after filter)")
print()

# ============================================================
# Part 3: 详细对比
# ============================================================
print("=" * 60)
print("Part 3: Detailed Comparison")
print("=" * 60)

# 找出 RapidOCR 有但 ocr_board 没有的
print("\n--- Missing from ocr_board (in RapidOCR but not board) ---")
missing_count = 0
for i, (rbox, rtext, rscore) in enumerate(rapid_boxes):
    # 检查是否有匹配的文本
    found = False
    for j, (btext, bscore) in enumerate(zip(board_texts, board_scores)):
        # 简单模糊匹配: 相同字符数且重叠度高
        if len(rtext) > 0 and len(btext) > 0:
            common = sum(1 for a, b in zip(rtext, btext) if a == b)
            overlap = common / max(len(rtext), len(btext))
            if overlap > 0.6:
                found = True
                # 显示分数差异
                score_diff = bscore - rscore
                if abs(score_diff) > 0.05 or rtext != btext:
                    print(f"  [{i}] SCORE DIFF: board={bscore:.3f} rapid={rscore:.3f} (diff={score_diff:+.3f})")
                    if rtext != btext:
                        print(f"       TEXT DIFF: rapid='{rtext}' board='{btext}'")
                break
    
    if not found and rtext.strip():
        missing_count += 1
        center = rbox.mean(axis=0)
        print(f"  [MISS #{missing_count}] '{rtext}' score={rscore:.3f} @ ({center[0]:.0f},{center[1]:.0f})")

# 找出 ocr_board 有但 RapidOCR 没有的
print("\n--- Extra in ocr_board (in board but not RapidOCR) ---")
extra_count = 0
for j, (btext, bscore, bbox) in enumerate(zip(board_texts, board_scores, board_boxes)):
    found = False
    for i, (rbox, rtext, rscore) in enumerate(rapid_boxes):
        if len(rtext) > 0 and len(btext) > 0:
            common = sum(1 for a, b in zip(rtext, btext) if a == b)
            overlap = common / max(len(rtext), len(btext))
            if overlap > 0.6:
                found = True
                break
    if not found:
        extra_count += 1
        center = bbox.mean(axis=0)
        print(f"  [EXTRA #{extra_count}] '{btext}' score={bscore:.3f} @ ({center[0]:.0f},{center[1]:.0f})")

# 分数差异统计
print("\n--- Score comparison for matched items ---")
score_diffs = []
for i, (rbox, rtext, rscore) in enumerate(rapid_boxes):
    for j, (btext, bscore, bbox) in enumerate(zip(board_texts, board_scores, board_boxes)):
        if len(rtext) > 0 and len(btext) > 0:
            common = sum(1 for a, b in zip(rtext, btext) if a == b)
            overlap = common / max(len(rtext), len(btext))
            if overlap > 0.6:
                score_diffs.append(bscore - rscore)
                break

if score_diffs:
    arr = np.array(score_diffs)
    print(f"  Matched pairs: {len(score_diffs)}")
    print(f"  Avg score diff (board-rapid): {arr.mean():+.4f}")
    print(f"  Min diff: {arr.min():+.4f}, Max diff: {arr.max():+.4f}")
    print(f"  Std: {arr.std():.4f}")

print(f"\nSummary: RapidOCR={len(rapid_boxes)}, Board={len(board_texts)}, "
      f"Missing={missing_count}, Extra={extra_count}")
