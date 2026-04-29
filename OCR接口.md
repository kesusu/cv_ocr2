# OCR 接口说明 — 怎么从别的py文件调用

> 文件: `ocr_workflow_accelerated.py`
> 跟 `asr.py` 的接口模式一样：**导入 → 调用/设回调 → 拿结果**

---

## 最简用法（只要这个就够了）

### 方式1：直接调用，拿文字结果

```python
# 你的主程序 / main.py / 任意py文件
from ocr_workflow_accelerated import recognize_image

text = recognize_image("photos/photo_1.jpg")   # 传入图片路径 或 numpy数组
print(text)
# 输出就是识别出来的纯文本，每行一段文字
```

### 方式2：设回调，识别完自动把结果传给你的处理函数

```python
# 你的主程序
from ocr_workflow_accelerated import set_ocr_callback, recognize_image

def handle_ocr_result(result, text):
    """OCR识别完会自动调这个函数"""
    # text        → 纯文本字符串（同 recognize_image 的返回值）
    # result      → 完整字典（含 texts/scores/count/avg_score/elapsed 等）
    print(f"收到OCR结果: {result['count']}段文字, 耗时{result['elapsed']:.1f}s")
    # ... 在这里做你的后续处理（发LLM、存数据库、显示UI等）...

set_ocr_callback(handle_ocr_result)   # 先注册回调
text = recognize_image("photo.jpg")   # 识别完自动触发 handle_ocr_result()
```

---

## 全部接口一览

| 函数 | 干嘛的 | 返回 |
|------|--------|------|
| **`recognize_image(path)`** | **识别一张图，返回纯文本** | `str` （每行一段） |
| `recognize_image_full(path)` | 同上，但返回完整信息(坐标/置信度/耗时) | `dict` |
| `init_ocr()` | 预加载模型（启动时调用一次，省去首次识别的0.17s） | 引擎实例 |
| `set_ocr_callback(fn)` | 设回调函数（识别完自动通知你） | 无 |
| `batch_recognize(paths)` | 批量识别多张图 | `list[dict]` |
| `get_ocr_status()` | 查引擎状态（是否已加载/模型名/耗时） | `dict` |

---

## 回调函数签名说明

跟 `asr.py` 的 `set_asr_callback` 一样：

```python
def your_function(result_dict, full_text):
    # result_dict: {
    #     'texts':     ['第一行文字', '第二行文字', ...],
    #     'scores':    [0.99, 0.97, ...],
    #     'count':     60,              # 文字段数
    #     'avg_score': 0.9613,         # 平均置信度
    #     'elapsed':   4.589,          # 总耗时(秒)
    # }
    # full_text:  "第一行文字\n第二行文字\n..."  (纯文本拼接)

set_ocr_callback(your_function)
```

---

## 集成示例：主程序启动 + OCR + 结果传给LLM处理

```python
# main.py  （你队友写的文件）
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ocr_workflow_accelerated import init_ocr, recognize_image, set_ocr_callback

# ---- 启动时预加载模型 ----
engine = init_ocr()   # 只需0.17s，之后每次识别不用重新加载

# ---- 设置回调：识别完自动处理 ----
def on_ocr_done(result, text):
    print(f"[main] OCR完成! {result['count']}段文字, 置信度{result['avg_score']:.2f}")
    # 把 text 发给 LLM 做问答 / 存入数据库 / 显示到界面 ...
    # send_to_llm(text)  
    # save_to_db(text)

set_ocr_callback(on_ocr_done)

# ---- 业务流程中调用 ----
def take_photo_and_recognize():
    # ... 拍照得到图片路径 photo_path ...
    text = recognize_image(photo_path)   # 识别完自动触发 on_ocr_done
    return text

if __name__ == "__main__":
    text = take_photo_and_recognize()
```

---

## 性能参考 (PC CPU)

| 项目 | 数值 |
|------|------|
| 模型加载（只需一次） | ~0.17s |
| 单张识别（药品说明书A4纸） | 3.5~5.5s |
| 平均置信度 | 0.94~0.96 |

## 依赖

```
pip install onnxruntime opencv-python numpy pyclipper shapely scikit-learn
```
