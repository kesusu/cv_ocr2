"""
integrate_beta.py - 异响检测与语音唤醒整合版（开发板版）
统一音频流：使用arecord统一采集，并行处理异响检测+VAD缓冲

========================================
【部署到不同环境时需修改的参数】
========================================

--- 必须修改（换机器/换模型时） ---
ONNX_MODEL_PATH  → ASR模型文件路径，更换模型或迁移时改为实际的.onnx路径
TOKENS_PATH      → 字符映射字典路径，与ONNX模型配套，换模型时同步更换tokens.txt
AUDIO_FOLDER     → 模式1批量测试的音频文件夹路径
WAKEWORDS        → 唤醒词集合，按需增删（4字词如"小光小光"识别率最高）
VAD_MODE         → VAD灵敏度(0~3)，噪声大→调高(2~3)，安静→调低(0~1)
ONE_SHOT_ASR     → True=唤醒后只转录一次再等重新唤醒；False=唤醒后持续转录
choice           → 启动模式："1"=批量测试，"2"=实时监听+唤醒

--- 异响检测阈值（一般不需改，误报/漏报时微调） ---
FREQUENCY_THRESHOLD          → 高低频分界线(Hz)，默认2500。拍手漏检可降低(如2200)，说话误报则提高
HIGH_FREQ_ENERGY_THRESHOLD   → 高频能量阈值，默认0.3。玻璃破碎漏检→降低(0.2)，底噪误报→提高(0.4~0.5)
IMPACT_CHANGE_RATE_THRESHOLD → 巨响变化率阈值，默认0.1。降低=更灵敏可能误报，提高=更保守可能漏报
ALARM_COOLDOWN               → 报警冷却时间(秒)，默认2.0。缩短=重复报警增多，延长=减少刷屏
MIN_VOL_ABS                  → 最小音量门槛，默认0.005。影响高频异响(玻璃破碎)的最小触发音量

--- 换ASR模型时需修改的代码区域（搜索 [ASR模块] 标记可快速定位） ---
[ASR模块-导入]      → 导入语句，换非ONNX模型需替换
[ASR模块-模型配置]  → ONNX_MODEL_PATH / TOKENS_PATH 路径
[ASR模块-模型加载类]→ OnnxModel 类，替换为新的模型加载方式
[ASR模块-特征预处理]→ cmvn / extract_mfcc_from_array / load_tokens，匹配新模型的特征格式
[ASR模块-初始化]    → VoiceWakeDetector.__init__ 中的模型加载代码
[ASR模块-核心识别方法]→ recognize_with_onnx 方法，重写推理+解码逻辑即可（接口：输入float32数组，输出文本字符串）

--- 情绪识别配置 ---
EMOTION_ENABLED       → 是否启用情绪识别（True/False），False时跳过情绪分析不消耗性能
EMOTION_CALLBACK      → 情绪结果回调函数格式: func(emotion_result, text)，通过 set_emotion_callback() 设置
SMART_EMOTION_PRELOAD → 模型加载时机（"1"=延迟加载启动快, "2"=启动预加载说命令零等待）

【智能情绪识别模式说明】
  - 默认状态: 仅使用文本关键词规则匹配（快速、零额外依赖）
  - 触发方式: 唤醒后说"打开智能情绪识别"等命令词
  - 智能模式: 文本关键词 + emotion2vec_plus_base 语音情绪融合
  - 融合逻辑: 语气识别(audio)第一答案 + 文本关键词(text)强匹配修正

【英文模式切换说明】
  - 默认语言: 中文(ONNX ASR)
  - 切换方式: 唤醒后说"切换英文"/"切换中文"等命令词（模糊匹配）
  - 英文引擎: fmodel Whisper tiny CPU（开发板专用，fiboaisdk SDK）
  - 中文引擎: ONNX TeleSpeech 模型
  - 英文回切: 英文模式下说 "back to chinese" / "switch to chinese" 等英文命令切回中文
  - WHISPER_PRELOAD → 模型加载时机（"1"=延迟加载启动快, "2"=启动预加载说命令零等待）
"""

import numpy as np
from scipy.fft import fft
import wave
import time
import os
from collections import deque
from datetime import datetime

# ============================================================
# 导入语音唤醒相关模块
# ============================================================
# ====== [ASR模块-导入] 换ASR模型时，以下导入可能需要替换 ======
import onnxruntime as ort        # ONNX推理引擎（换非ONNX模型可删）
import webrtcvad                 # VAD语音活动检测（保留，与ASR无关）
import subprocess                # 子进程调用arecord录音（替代pyaudio，适配Linux开发板）
import select                    # 非阻塞管道读取（配合subprocess使用）
import soundfile as sf           # 音频文件读写（换ASR模型可能需要/替换）
from difflib import SequenceMatcher  # 唤醒词模糊匹配（保留，与ASR无关）
import kaldi_native_fbank as knf # MFCC特征提取（换ASR模型时可能替换为librosa等）
# ====== [英文模块-导入] 英文识别使用fmodel Whisper（开发板专用）=====
# 注意：fiboaisdk 仅在开发板上可用，延迟导入避免PC端报错
# ====== [英文模块-导入结束] ================================
# ====== [ASR模块-导入结束] ===================================

# ====== [情绪模块-导入] 换SER模型时，以下导入需要替换 ======
import json  # 用于加载情绪关键词规则JSON文件
import logging
import threading
logging.getLogger("modelscope").setLevel(logging.WARNING)
logging.getLogger("funasr").setLevel(logging.WARNING)
# ====== [情绪模块-导入结束] ===================================


# ===================== 情绪 -> ID/Emoji 映射表（与emotion2vec_plus_base对齐）=====================
# emotion2vec输出格式: "中文/english"，如 "生气/angry"
EMOTION_EMOJI_MAP = {
    "生气/angry":    {"emoji": "😠", "id": 0, "key": "angry",     "cn": "愤怒"},
    "厌恶/disgusted": {"emoji": "🤢", "id": 1, "key": "disgusted", "cn": "厌恶"},
    "恐惧/fearful":  {"emoji": "😨", "id": 2, "key": "fearful",   "cn": "恐惧"},
    "开心/happy":    {"emoji": "😊", "id": 3, "key": "happy",     "cn": "开心"},
    "中立/neutral":  {"emoji": "😐", "id": 4, "key": "neutral",   "cn": "中立"},
    "其他/other":    {"emoji": "🤔", "id": 5, "key": "other",     "cn": "其他"},
    "难过/sad":      {"emoji": "😢", "id": 6, "key": "sad",       "cn": "伤心"},
    "吃惊/surprised": {"emoji": "😲", "id": 7, "key": "surprised", "cn": "惊讶"},
}
DEFAULT_EMOJI_INFO = {"emoji": "😐", "id": 4, "key": "neutral", "cn": "中立"}

def get_emotion_info(emotion_str):
    """根据emotion2vec返回的情绪字符串获取统一格式信息"""
    return EMOTION_EMOJI_MAP.get(emotion_str, DEFAULT_EMOJI_INFO)

# ==================== 统一参数配置 ====================
SAMPLING_RATE = 16000
CHUNK_SIZE = 480  # 30ms 音频帧 (480 samples)，满足VAD要求
ALARM_COOLDOWN = 2.0       # 异响报警冷却时间

# ==================== 录音设备配置（开发板适配）====================
# arecord设备名（Linux开发板），可通过 `arecord -l` 查看可用设备
# 常见格式: "plughw:CARD=Device,DEV=0" 或 "default" 或 "hw:0,0"
ARECORD_DEVICE = "plughw:CARD=Device,DEV=0"
ARECORD_CHANNELS = 1
ARECORD_FORMAT = "S16_LE"   # 16bit signed little-endian（与pyaudio paInt16一致）
# ===============================================

# ==================== ASR模式配置 ====================
ONE_SHOT_ASR = True   # True: 唤醒后只转录一次，然后回到等待唤醒状态；False: 持续对话模式

# --- 动态阈值参数 ---
BASELINE_WINDOW = 5.0       # 基线计算窗口(秒)
BASELINE_UPDATE_RATE = 0.2  # 基线更新速率
MIN_VOL_ABS = 0.005        # 适应更小音量的尖锐声音

# --- 优化后的特征阈值 ---
FREQUENCY_THRESHOLD = 2500  # 频谱质心阈值（高低频分界线）
HIGH_FREQ_ENERGY_THRESHOLD = 0.3  # 高频能量阈值（降低以适应实际计算值）
IMPACT_CHANGE_RATE_THRESHOLD = 0.1  # 音量变化率阈值
# ===============================================

# ============================================================
# 唤醒词相关参数（来自asr_wake.py，保持原样）
# ============================================================
VAD_MODE = 1  # VAD灵敏度(0=极严格, 1=严格, 2=中等, 3=极宽松)，调低可过滤远处人声/环境噪声

# 唤醒词列表（目标唤醒词）
WAKEWORDS = {"小光小光", "小光你好", "你好小光", "打开小光"}
# 小光你好基本不成功，打开小光一般，其余准确率高（对于whisper_tiny）

# 智能情绪识别命令词（语音触发切换到情绪增强模式）
SMART_EMOTION_COMMANDS = {
    "打开智能情绪识别", "开启智能情绪识别", "启动智能情绪识别",
    "智能情绪识别", "进入智能情绪", "开始情绪识别",
}

# 英文模式切换命令词（语音触发切换到英文ASR）
ENGLISH_MODE_COMMANDS = {
    "切换英文", "切换英语", "换成英文", "换成英语",
    "英文模式", "英语模式", "说英文", "说英语",
}

# 切换回中文模式的命令词（中文模式触发）
CHINESE_MODE_COMMANDS = {
    "切换中文", "切换汉语", "换成中文", "换成汉语",
    "中文模式", "汉语模式", "说中文", "说汉语",
}

# 切换回中文模式的英文命令词（英文模式下用这些回切，不易在日常对话中误触）
BACK_TO_CHINESE_EN_COMMANDS = {
    "back to chinese",
    "switch to chinese",
    "change to chinese",
    "chinese mode please",
    "go back to chinese",
}

# 英文模式(fmodel Whisper)模型加载时机配置（与智能情绪识别相同的策略）:
WHISPER_PRELOAD = "1"    # "1"=延迟加载启动快, "2"=启动预加载说命令零等待

# ====== [ASR模块-模型配置] 换ASR模型时修改此处路径和配套文件 ======
# ONNX ASR 模型路径（换模型改为新的.onnx文件路径）
ONNX_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.int8.onnx")
# 字符映射字典路径（与ONNX模型配套，换模型需同步更换对应的tokens/词表文件）
TOKENS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.txt")
# ====== [ASR模块-模型配置结束] ===================================

# ====== [情绪模块-配置] 情绪识别相关参数 ======
EMOTION_ENABLED = True          # 是否启用情绪识别（False时完全跳过，零开销）
EMOTION_SER_MODEL_PATH = ""     # SER情绪模型路径（当前使用funasr emotion2vec_plus_base，自动从ModelScope下载）

# 智能情绪识别模式配置:
#   SMART_EMOTION_PRELOAD: 模型何时加载
#     "1" = 延迟加载：启动快，说"打开智能情绪识别"时才加载模型(需等几秒)
#     "2" = 启动预加载：启动时即加载模型(慢几秒)，说命令后零等待切换
SMART_EMOTION_PRELOAD = "1"
# ====== [情绪模块-配置结束] ===================================

# ============================================================
# fmodel Whisper 开发板专用配置（参照 asr_whisper_fibo.py）
#   使用 fiboaisdk SDK: api_audio_py.AudioAPI + license_py
#   转录方式: 传入WAV文件路径（非原始PCM）
# ============================================================
FMODEL_LICENSE_DIR = "/home/fibo/asr/qcom_6490_license"
FMODEL_MODEL_FILE = "/home/fibo/asr/whisper_tiny_cpu_1.0.0_onnx_all_0bc78d5f7ef9f78362960cbf5ca755fc.fmodel"


# ====== [ASR模块-模型加载类] 换ASR模型时替换整个类或修改推理调用方式 ======
class OnnxModel:
    """ONNX推理模型封装类（来自 test_time.py）"""
    def __init__(self, filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        self.model = ort.InferenceSession(
            filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, x):
        logits = self.model.run(
            [self.model.get_outputs()[0].name],
            {self.model.get_inputs()[0].name: x},
        )[0]
        return logits
# ====== [ASR模块-模型加载类结束] ===================================


# ====== [ASR模块-特征预处理] 换ASR模型时可能需要替换预处理函数（如换用不同特征） ======
def cmvn(features):
    """Cepstral Mean and Variance Normalization（倒谱均值方差归一化）"""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + 1e-5)


def extract_mfcc_from_array(audio_float):
    """
    从float32 numpy数组中提取MFCC特征（40维）
    与test_time.py的get_features逻辑一致，但直接从内存数组读取
    
    Args:
        audio_float: float32 numpy数组，采样率16000Hz
    Returns:
        mfcc_features: MFCC特征矩阵 (T, 40)，未做CMVN
    """
    # 缩放因子（与test_time.py一致）
    samples = audio_float * 372768

    opts = knf.MfccOptions()
    opts.frame_opts.dither = 0
    opts.num_ceps = 40
    opts.use_energy = False
    opts.mel_opts.num_bins = 40
    opts.mel_opts.low_freq = 40
    opts.mel_opts.high_freq = -200

    mfcc = knf.OnlineMfcc(opts)
    mfcc.accept_waveform(SAMPLING_RATE, samples)
    frames = []
    for i in range(mfcc.num_frames_ready):
        frames.append(mfcc.get_frame(i))

    return np.stack(frames, axis=0)


def load_tokens(tokens_path):
    """加载字符映射字典（换ASR模型时需匹配新模型的词表格式）"""
    id2token = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t
    return id2token
# ====== [ASR模块-特征预处理结束] ===================================


# ====== [情绪模块-核心类] 来自 emo+en_pc.py（完整移植）=====================
class EmotionDetector:
    """
    情绪识别检测器（增强版：文本关键词 + 语音情绪融合）
    
    两种模式：
      - 普通模式(默认): 仅使用文本关键词规则匹配（零额外依赖，快速）
      - 智能模式: 文本关键词 + emotion2vec_plus_base 语音情绪识别融合
    
    【智能模式融合逻辑】
      第1步: 语气识别(audio) → 得到第一答案
      第2步: 文本关键词(text) → 若命中强关键词则修正/覆盖第1答案
      第3步: 输出最终融合结果
    
    【8类情绪分类（与SER模型一致）】
      0=愤怒(angry)  1=厌恶(disgusted)  2=恐惧(fearful)  3=开心(happy)
      4=中立(neutral) 5=其他(other)     6=伤心(sad)      7=惊讶(surprised)
    """
    
    def __init__(self, ser_model_path=None, rules_path=None, preload=False):
        self._rules_path = rules_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "emotion_rules.json")
        
        # 加载文本关键词规则
        with open(self._rules_path, encoding='utf-8') as f:
            all_data = json.load(f)
        self.rules = {k: v for k, v in all_data.items() if not k.startswith('_')}
        
        total_kw = sum(len(v['keywords']) for v in self.rules.values())
        emotion_names = [f"{v['emoji']} {v['label_cn']}" for v in sorted(self.rules.values(), key=lambda x: x['id'])]
        print(f"📂 情绪规则已加载: {os.path.basename(self._rules_path)} ({len(self.rules)}类/{total_kw}词)")
        print(f"   {' | '.join(emotion_names)}")
        
        # ====== emotion2vec_plus_base 语音情绪模型（轻量版） ======
        self.ser_model = None
        self._ser_model_name = "iic/emotion2vec_plus_base"
        self._ser_loaded = False
        self._temp_audio_counter = 0
        
        if preload:
            # 预加载模式：启动时直接加载模型
            self._load_ser_model()
        else:
            print(f"📌 智能情绪模式: 待激活（首次触发\"打开智能情绪识别\"时加载模型）")
    
    def _load_ser_model(self):
        """加载emotion2vec模型（支持预加载和延迟加载两种模式）"""
        if self._ser_loaded:
            return True
        
        # 先完成import（不计入模型加载时间）
        from funasr import AutoModel
        
        import io, sys
        print("🔄 [情绪] 正在加载 emotion2vec_plus_base 语音情绪模型...")
        load_start = time.time()
        
        # 屏蔽冗余日志
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # 优先从本地加载（离线部署：把 emotion_model_cache 文件夹放在 asr.py 同目录下）
            _local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_model_cache", self._ser_model_name)
            _model_path = _local_model_path if os.path.exists(_local_model_path) else self._ser_model_name
            self.ser_model = AutoModel(
                model=_model_path,
                disable_update=True,
                device="cpu",
                verbose=False
            )
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            load_time = time.time() - load_start
            self._ser_loaded = True
            print(f"✅ [情绪] 语音情绪模型已加载！耗时: {load_time:.1f}s")
            return True
        except Exception as e:
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            print(f"❌ 语音情绪模型加载失败: {e}")
            print("   回退到纯文本模式")
            self._ser_loaded = False
            return False
    
    def detect_from_text(self, text):
        """从文本中识别情绪（关键词匹配，8类）"""
        if not text or not text.strip():
            return None
        
        text_clean = text.strip()
        best_match = None
        best_score = 0
        
        for category, rule in self.rules.items():
            matched = [kw for kw in rule['keywords'] if kw in text_clean]
            score = len(matched)
            
            if score > best_score:
                best_score = score
                best_match = {
                    'emotion': category,
                    'emotion_id': rule['id'],
                    'label_cn': rule['label_cn'],
                    'label_en': rule['label_en'],
                    'confidence': min(score / max(len(text_clean) * 0.1, 1), 1.0),
                    'emoji': rule['emoji'],
                    'matched_keywords': matched,
                    'method': 'text_rule',
                }
        
        if best_match is None:
            return {
                'emotion': 'neutral', 'emotion_id': 4,
                'label_cn': '中立', 'label_en': 'neutral',
                'confidence': 0.3, 'emoji': '😐',
                'matched_keywords': [], 'method': 'text_rule_default_neutral',
            }
        
        return best_match
    
    def detect_from_audio(self, audio_float):
        """
        从音频float32数组中识别情绪（emotion2vec模型）
        
        Args:
            audio_float: float32 numpy数组，采样率16000Hz
        
        Returns:
            dict 或 None（模型未加载或推理失败时）
        """
        if not self._load_ser_model() or self.ser_model is None:
            return None
        
        try:
            import soundfile as sf
            import io, sys
            
            # 写入临时WAV文件（funasr需要文件路径输入）
            self._temp_audio_counter += 1
            temp_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"_temp_ser_{self._temp_audio_counter}.wav"
            )
            sf.write(temp_path, audio_float, SAMPLING_RATE)
            
            # 调用emotion2vec推理（屏蔽tqdm等冗余输出）
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                res = self.ser_model.generate(
                    temp_path,
                    output_dir="./outputs",
                    granularity="utterance",
                    extract_embedding=False
                )
            finally:
                sys.stdout, sys.stderr = orig_stdout, orig_stderr
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # 解析结果
            if res and len(res) > 0:
                result = res[0]
                if isinstance(result, dict):
                    labels = result.get("labels", [])
                    scores = result.get("scores", [])
                    
                    if scores:
                        max_idx = scores.index(max(scores))
                        top_emotion = labels[max_idx] if max_idx < len(labels) else "未知"
                        top_score = scores[max_idx]
                        
                        info = get_emotion_info(top_emotion)
                        return {
                            'emotion': info['key'],
                            'emotion_id': info['id'],
                            'label_cn': info['cn'],
                            'label_en': info['key'],
                            'confidence': float(top_score),
                            'emoji': info['emoji'],
                            'matched_keywords': [],
                            'method': 'speech_ser',
                            'all_labels': labels,
                            'all_scores': scores,
                        }
            
            return None
        except Exception as e:
            print(f"   [SER推理异常] {e}")
            return None
    
    def detect_smart(self, text, audio_float=None):
        """
        智能融合模式：语气识别第一答案 + 文本关键词修正
        
        融合策略：
          1. emotion2vec(音频) → 主答案
          2. 文本关键词匹配 → 若命中且置信度高 → 修正主答案
          3. 若仅一方有结果 → 直接采用该方结果
        
        Returns:
            dict: 最终情绪结果
        """
        # Step 1: 语气识别作为第一答案
        audio_result = None
        if audio_float is not None and len(audio_float) > 0:
            audio_result = self.detect_from_audio(audio_float)
        
        # Step 2: 文本关键词匹配
        text_result = self.detect_from_text(text)
        
        # --- 融合判断 ---
        if audio_result is None:
            # 语音模型不可用/失败 → 纯文本兜底
            final = dict(text_result)
            final['method'] = 'smart_fallback_text'
            return final
        
        if text_result is None:
            # 文本为空 → 纯语音结果
            final = dict(audio_result)
            final['method'] = 'smart_fallback_audio'
            return final
        
        # 两方都有结果 → 融合决策
        text_confidence = text_result.get('confidence', 0)
        text_has_strong_kw = len(text_result.get('matched_keywords', [])) >= 2
        text_emotion_key = text_result.get('emotion', '')
        
        # 修正条件：文本命中了>=2个关键词，说明有明确语义倾向
        if text_has_strong_kw and text_confidence > 0.3:
            # 用文本结果覆盖，但保留音频的置信度参考
            final = dict(text_result)
            final['method'] = 'smart_text_corrected'
            final['_audio_origin'] = {
                'emotion': audio_result['emotion'],
                'confidence': audio_result['confidence'],
            }
        else:
            # 采用语音结果作为主答案
            final = dict(audio_result)
            final['method'] = 'smart_audio_primary'
            final['_text_ref'] = {
                'emotion': text_result['emotion'],
                'confidence': text_result['confidence'],
            }
        
        return final
    
    def detect(self, text, audio_float=None, smart_mode=False):
        """
        统一情绪识别入口
        
        Args:
            text: ASR转录文本
            audio_float: 音频float32数组（可选）
            smart_mode: 是否启用智能融合模式
        
        Returns:
            dict: 情绪结果
        """
        if smart_mode:
            return self.detect_smart(text, audio_float)
        else:
            return self.detect_from_text(text)
    
    def format_result(self, result):
        """将情绪结果格式化为可读字符串"""
        if result is None:
            return ""
        method_tag = ""
        method = result.get('method', '')
        if 'smart' in method:
            mode_map = {
                'smart_audio_primary': '🎤️语音优先',
                'smart_text_corrected': '📝文本修正',
                'smart_fallback_text': '📝文本兜底',
                'smart_fallback_audio': '🎤️语音兜底',
            }
            method_tag = f" [{mode_map.get(method, method)}]"
        elif method == 'speech_ser':
            method_tag = " [语音SER]"
        elif 'text' in method:
            method_tag = " [文本]"
        
        return f"{result['emoji']} [{result['label_cn']}] ID:{result.get('emotion_id','?')} 置信度:{result['confidence']:.2f}{method_tag}"
# ====== [情绪模块-核心类结束] ===================================


"""
================================================================================
【报警触发逻辑概括】

1. 🔨 低频巨响 (针对：凳子摔倒、重物撞击)
   - 音量 (RMS)：超过动态基线的 1.5 倍 + 0.02 偏移量（瞬间能量突变）。
   - 高频能量：必须大于 0.3（伴随宽频带的冲击声）。
   - 变化率：音量上升坡度陡峭，具备明显的冲击性。
   - 质心条件：≤ 2500Hz（低频为主）

2. 📢 高频巨响 (针对：拍手等高频大音量冲击)
   - 与低频巨响条件相同，但质心 > 2500Hz（高频为主）

3. 🔪 高频异响 (针对：玻璃破碎、尖锐摩擦、金属碰撞)
   - 频谱质心：重心超过 2500Hz（声音听觉上非常尖锐）。
   - 高频能量：3kHz-8kHz 频段的能量均值超过 0.3。
   - 最小音量：只需超过 0.005 (MIN_VOL_ABS)，不依赖动态基线（适应小声但刺耳的异响）

判断优先级：巨响 > 异响（满足巨响条件优先判定为巨响）
================================================================================
"""


class SmartMonitor:
    """异响检测监控器（与 integrate_beta copy.py 实时模式一致）"""
    def __init__(self):
        self.last_alarm_time = 0
        self.baseline_vol = 0.01
        # 使用固定的 vol_history 容量，与原文件保持一致
        self.vol_history = deque(maxlen=int(BASELINE_WINDOW * SAMPLING_RATE / 1024))
        self.last_vol = 0.0

    def calculate_features(self, audio_data):
        if len(audio_data) == 0: return 0.0, 0.0, 0.0, 0.0
        rms = np.sqrt(np.mean(audio_data**2))
        
        # 直接用原始数据进行FFT（与原文件一致）
        window = np.hamming(len(audio_data))
        yf = fft(audio_data * window)
        mag = np.abs(yf[:len(audio_data)//2])
        freqs = np.arange(len(mag)) * SAMPLING_RATE / (2 * len(mag))
        
        if np.sum(mag) > 0:
            centroid = np.sum(freqs * mag) / np.sum(mag)
        else:
            centroid = 0.0
            
        high_freq_mask = (freqs >= 3000) & (freqs <= 8000)
        high_freq_energy = np.mean(mag[high_freq_mask]) if np.any(high_freq_mask) else 0.0
        return rms, centroid, high_freq_energy, mag

    def update_baseline(self, current_vol):
        """更新动态基线"""
        self.vol_history.append(current_vol)
        if len(self.vol_history) > 0:
            avg_vol = np.mean(self.vol_history)
            self.baseline_vol = self.baseline_vol * (1 - BASELINE_UPDATE_RATE) + avg_vol * BASELINE_UPDATE_RATE
            self.baseline_vol = max(self.baseline_vol, MIN_VOL_ABS)

    def calculate_change_rate(self, current_vol):
        """计算音量变化率"""
        if self.last_vol == 0.0:
            change_rate = 0.0
        else:
            change_rate = abs(current_vol - self.last_vol) / (CHUNK_SIZE / SAMPLING_RATE)
        self.last_vol = current_vol
        return change_rate

    def process_audio(self, audio_data, silent=False):
        """处理音频数据进行异响检测（与原文件 integrate_beta copy.py 实时模式一致）
        
        Args:
            audio_data: 音频数据
            silent: True时跳过调试打印（唤醒后使用）
        """
        current_vol, centroid, high_freq_energy, mag = self.calculate_features(audio_data)
        
        # 更新基线
        self.update_baseline(current_vol)
        
        # 计算音量变化率
        change_rate = self.calculate_change_rate(current_vol)
        
        # --- 与原文件 integrate_beta copy.py 实时模式一致的判断逻辑 ---
        
        is_alarm = False
        alarm_type = ""
        
        # 低频巨响：音量突变 + 高频能高 + 变化率大
        is_impact = (current_vol > self.baseline_vol * 1.5 + 0.02) and \
                    (high_freq_energy > HIGH_FREQ_ENERGY_THRESHOLD) and \
                    (change_rate > IMPACT_CHANGE_RATE_THRESHOLD)
        
        # 高频异响：质心高 + 高频能高 + 音量超过最小阈值
        is_glass = (centroid > FREQUENCY_THRESHOLD) and \
                  (high_freq_energy > HIGH_FREQ_ENERGY_THRESHOLD) and \
                  (current_vol > MIN_VOL_ABS)
        
        if is_impact:
            freq_type = "高频" if centroid > FREQUENCY_THRESHOLD else "低频"
            alarm_type = f"{freq_type}巨响"
            is_alarm = True
        elif is_glass:
            alarm_type = "高频异响"
            is_alarm = True
        
        # --- 触发报警 ---
        if is_alarm:
            self.trigger_alarm(alarm_type, current_vol, centroid, high_freq_energy)

        # 调试打印（唤醒后silent=True时不打印）
        if not silent:
            print(f"\r[音量:{current_vol:.3f}] [质心:{centroid:.0f}Hz] [高频能:{high_freq_energy:.2f}]      ", end="")

    def trigger_alarm(self, alarm_type, actual_volume, centroid, high_freq_energy):
        now = time.time()
        if now - self.last_alarm_time < ALARM_COOLDOWN:
            return
        self.last_alarm_time = now
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n\n🚨 【{alarm_type}报警】")
        print(f"   ⏰ 时间: {current_time_str}")
        print(f"   🔍 类型: {alarm_type}")
        print(f"   🔊 音量: {actual_volume:.3f}")
        print(f"   📊 质心: {centroid:.0f}Hz")
        print(f"   📈 高频能: {high_freq_energy:.1f}")
        print(f"   🔧 基线: {self.baseline_vol:.4f}")
        print("-" * 40)


class VoiceWakeDetector:
    """语音唤醒检测器（来自asr_wake.py，保持原样 + 新增情绪/英文功能）"""
    def __init__(self):
        # ====== [ASR模块-初始化] 换ASR模型时修改模型加载和字典加载方式 ======
        # ---- 加载 ONNX ASR 模型（替代原 Whisper）----
        print("🔄 [ASR] 正在加载 TeleSpeech ONNX 模型...")
        load_start = time.time()

        self.model = OnnxModel(ONNX_MODEL_PATH)
        self.id2token = load_tokens(TOKENS_PATH)

        asr_load_time = time.time() - load_start
        print(f"✅ [ASR] 模型加载完成！耗时: {asr_load_time:.3f} 秒")
        # ====== [ASR模块-初始化结束] ===================================

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(VAD_MODE)  # 设置为中等宽松模式
        
        self.chunk_size = CHUNK_SIZE
        self.is_awake = False  # 唤醒状态标志
        self.wake_callback = None
        self.one_shot_done = False  # ONE_SHOT_ASR模式下是否已完成一次转录
        
        # 音频缓冲区（用于累积语音片段）
        self.audio_buffer = []
        
        # 首次识别标记（用于区分预热/正常耗时）
        self.asr_first_run = True

        # LLM处理期间暂停标志（避免终端信息过多+节省CPU）
        self._paused = False

        # ---- 尾静默截断优化（减少等待VAD判静音的时间）----
        self.trailing_silent_frames = 0      # 连续静音帧计数
        self.TRAILING_SILENT_MAX = 7         # 尾部连续多少帧(30ms/帧)静音后强制处理（7帧≈210ms，满足0.2s内响应）
        self.MAX_SPEECH_FRAMES = 100         # 最大语音缓冲帧数（100帧≈3秒），超限强制处理（防VAD一直误判语音）
        self.MIN_SPEECH_FOR_FLUSH = 15       # 最少缓冲帧数(≈450ms)，低于此值不强制处理（避免识别碎片）

        # ====== [情绪模块-初始化] 与 emo+en_pc.py 完全一致 ======
        if EMOTION_ENABLED:
            print("🔄 正在加载情绪识别模块...")
            _preload_model = (SMART_EMOTION_PRELOAD == "2")
            if _preload_model:
                print("   [模型预加载] 启动时加载emotion2vec模型（说命令后零等待切换）")
            else:
                print("   [延迟加载] 模型将在说\"打开智能情绪识别\"时才加载")
            self.emotion_detector = EmotionDetector(
                ser_model_path=EMOTION_SER_MODEL_PATH if EMOTION_SER_MODEL_PATH else None,
                preload=_preload_model
            )
            self.emotion_enabled = True
            self.last_emotion_result = None
            self._emotion_callback = None
            # 无论是否预加载，默认都不开启智能模式（需语音命令触发）
            self.smart_emotion_mode = False
        else:
            self.emotion_detector = None
            self.emotion_enabled = False
            self.last_emotion_result = None
            self.smart_emotion_mode = False
        # ====== [情绪模块-初始化结束] ===================================

        # ====== [英文模块-初始化] 英文识别使用fmodel Whisper（开发板专用）=====
        self.english_mode = False       # 默认中文模式(ONNX)
        self.whisper_model = None         # (au_api, api_instance) 元组
        self.whisper_loaded = False       # 是否已加载
        
        _whisper_preload = (WHISPER_PRELOAD == "2")
        if _whisper_preload:
            print("🔄 [EN] 启动时预加载 fmodel Whisper 模型...")
            self._load_whisper_model()
        else:
            print("   [延迟加载] fmodel Whisper将在说\"切换英文\"时才加载")
        # ====== [英文模块-初始化结束] ======================

        # ---- 临时WAV计数器（供fmodel和emotion2vec使用）----
        self._temp_wav_counter = 0
        self._temp_wav_lock = threading.Lock()

    # ----------------------------------------------------------
    # fmodel Whisper 加载（开发板专用，参照 asr_whisper_fibo.py）
    # ----------------------------------------------------------
    def _load_whisper_model(self):
        """加载 fmodel Whisper 模型（开发板专用）"""
        if self.whisper_loaded and self.whisper_model is not None:
            return True

        try:
            from fiboaisdk.api_aisdk_py import api_audio_py as au_api
            from fiboaisdk.api_aisdk_py import license_py as license_api
            t0 = time.perf_counter()

            # 1. 初始化许可证
            print(f"   正在初始化许可证 ({FMODEL_LICENSE_DIR})...")
            lic_ok = self._init_fmodel_license(license_api)
            if not lic_ok:
                print("   ⚠️ 许可证初始化失败，但继续尝试加载模型")

            # 2. 加载 .fmodel 模型
            model_file = FMODEL_MODEL_FILE
            if not os.path.exists(model_file):
                print(f"   ⚠️ fmodel模型文件不存在: {model_file}")
                return False

            print(f"   正在加载 fmodel: {os.path.basename(model_file)} ...")
            api_instance = au_api.AudioAPI()
            api_instance.Init(model_file)

            self.whisper_model = (au_api, api_instance)
            self.whisper_loaded = True

            t_load = time.perf_counter() - t0
            print(f"   ✓ fmodel Whisper 加载完成 ({t_load:.1f}s)")
            print(f"      模型: {model_file}")
            return True

        except ImportError:
            print("   ⚠️ 未安装 fiboaisdk 库（开发板专用SDK不可用）")
            return False
        except Exception as e:
            print(f"   ⚠️ fmodel Whisper 加载失败: {e}")
            return False

    @staticmethod
    def _init_fmodel_license(license_api):
        """初始化 fmodel 许可证（参照 asr_whisper_fibo.py）"""
        lic_dir = FMODEL_LICENSE_DIR
        if not os.path.exists(lic_dir):
            print(f"   ⚠️ 许可证目录不存在: {lic_dir}")
            return False

        def _read_file(path):
            try:
                mode = "r" if path.endswith(".pem") else "rb"
                with open(path, mode) as f:
                    return f.read()
            except Exception:
                return None

        key1 = _read_file(os.path.join(lic_dir, "key1.pem"))
        key2 = _read_file(os.path.join(lic_dir, "key2.pem"))
        key3 = _read_file(os.path.join(lic_dir, "key3.pem"))
        candidates = [os.path.join(lic_dir, "license1.bin"),
                      os.path.join(lic_dir, "license.bin")]
        license_data = None
        for c in candidates:
            if os.path.exists(c):
                license_data = _read_file(c)
                break

        if all([key1, key2, key3, license_data]):
            ret = license_api.Init(key1, key2, key3, license_data)
            print(f"   许可证初始化返回值: {ret}")
            return ret == 0

        print("   ⚠️ 许可证文件不完整")
        return False

    def _whisper_transcribe(self, audio_np_int16):
        """使用 fmodel 进行 Whisper 转录（开发板专用）
        
        参照 asr_whisper_fibo.py 的 transcribe_audio 流程:
          1. 将音频保存为临时WAV文件
          2. 创建 FiboAudio 对象设置 audio_path
          3. 调用 TranscribeSync 同步转录
          4. 从 result.speech_text 取结果
        """
        if not self.whisper_loaded or self.whisper_model is None:
            return "[Whisper未加载]"

        try:
            au_api, api_instance = self.whisper_model

            # 1. 保存临时WAV文件（fmodel需要WAV文件路径输入）
            wav_path = self._create_temp_wav(audio_np_int16)
            if not wav_path or not os.path.exists(wav_path):
                return ""

            # 2. 构建FiboAudio对象（参照 asr_whisper_fibo.py line 289-294）
            audio = au_api.FiboAudio()
            audio.audio_sample_rate = 16000
            audio.audio_channel = 1
            audio.audio_format = au_api.FiboAudioFormat.FIBO_AUDIO_FORMAT_WAV
            audio.audio_path = wav_path
            audio.extra_params = ""

            # 3. 调用 TranscribeSync 同步转录
            result = au_api.ResultNlpAudio()
            api_instance.TranscribeSync(audio, result, 10)

            # 4. 提取结果
            text = ""
            if hasattr(result, "speech_text"):
                text = str(result.speech_text).strip()

            # 清理临时文件
            self._cleanup_temp_wav(wav_path)

            return text

        except Exception as e:
            print(f"   ⚠️ fmodel Whisper 推理错误: {e}")
            return ""

    # ----------------------------------------------------------
    # 临时WAV文件管理
    # ----------------------------------------------------------
    def _create_temp_wav(self, audio_np_int16):
        """保存临时WAV文件"""
        try:
            with self._temp_wav_lock:
                self._temp_wav_counter += 1
                counter = self._temp_wav_counter
            fname = f"_temp_asr_{counter}.wav"
            fpath = os.path.abspath(fname)
            with wave.open(fpath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLING_RATE)
                wf.writeframes(audio_np_int16.tobytes())
            return fpath
        except Exception as e:
            print(f"   [调试] WAV保存失败: {e}")
            return ""

    @staticmethod
    def _cleanup_temp_wav(fpath):
        """清理临时文件"""
        if fpath and os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    # ----------------------------------------------------------
    # 原有接口：唤醒词模糊匹配 + VAD + ASR
    # ----------------------------------------------------------
    def is_speech_detected(self, frame):
        """使用 VAD 检测语音活动"""
        return self.vad.is_speech(frame, SAMPLING_RATE)

    def is_fuzzy_match(self, recognized_text):
        """
        模糊匹配唤醒词（宽松版，补偿ASR切片/模型不完美）
        
        匹配优先级：
        1. 完全匹配：识别结果完全等于某个唤醒词
        2. 子串包含：唤醒词的任意连续2字片段出现在结果中（如"小光"在"光小包"中）
        3. 编辑距离：相似度 >= 0.5（比原来0.7更宽松）
        """
        if recognized_text in WAKEWORDS:
            return True
        
        # ---- 补偿策略1: 唤醒词片段子串匹配 ----
        # 提取所有唤醒词中的2字连续片段，检查是否出现在识别结果中
        for wakeword in WAKEWORDS:
            # 取2字片段（覆盖"小光""你好""打开"等核心词组）
            for i in range(len(wakeword) - 1):
                fragment = wakeword[i:i+2]
                if fragment in recognized_text and len(fragment) == 2:
                    print(f"片段匹配成功: \"{fragment}\" ∈ \"{recognized_text}\" (来自唤醒词: {wakeword})")
                    return True
        
        # ---- 补偿策略2: 编辑距离相似度（阈值从0.7降到0.5）----
        for wakeword in WAKEWORDS:
            ratio = SequenceMatcher(None, recognized_text, wakeword).ratio()
            if ratio >= 0.5:
                print(f"模糊匹配成功: {recognized_text} ≈ {wakeword} (相似度: {ratio:.2f})")
                return True
        
        return False

    def is_command_match(self, recognized_text, command_set):
        """
        模糊匹配命令词（与唤醒词相同的宽松策略，提高命令识别率）
        
        Args:
            recognized_text: ASR识别文本
            command_set: 命令词集合（如 SMART_EMOTION_COMMANDS）
        
        Returns:
            bool: 是否命中命令
        """
        if not recognized_text:
            return False
        
        text = recognized_text.strip()
        
        # 检测命令集是否为英文（用于选择匹配策略）
        is_english_cmds = any(not all('\u4e00' <= c <= '\u9fff' or c == ' ' for c in cmd) 
                               for cmd in command_set if cmd)
        
        # 1. 完全匹配（大小写不敏感，适用于英文）
        text_lower = text.lower()
        if any(text_lower == cmd.lower() for cmd in command_set):
            return True
        
        if not is_english_cmds:
            # 中文命令集：使用2字符片段匹配（中文友好）
            for cmd in command_set:
                for i in range(len(cmd) - 1):
                    fragment = cmd[i:i+2]
                    if fragment in text and len(fragment) == 2:
                        print(f"命令片段匹配: \"{fragment}\" ∈ \"{text}\" (来自命令: {cmd})")
                        return True
        else:
            # 英文命令集：使用单词级片段匹配（避免 "o " 等单字符误触）
            for cmd in command_set:
                cmd_lower = cmd.lower()
                words = cmd_lower.split()
                for i in range(len(words)):
                    # 取连续2个单词作为片段
                    frag = ' '.join(words[i:i+2])
                    if len(frag) >= 4 and frag in text_lower:
                        print(f"命令单词片段匹配: \"{frag}\" ∈ \"{text}\" (来自命令: {cmd})")
                        return True
                    # 单个长词(>=4字符)也做子串匹配
                    single_word = words[i]
                    if len(single_word) >= 4 and single_word in text_lower:
                        print(f"命令单词匹配: \"{single_word}\" ∈ \"{text}\" (来自命令: {cmd})")
                        return True
        
        # 3. 编辑距离相似度 >= 0.5
        for cmd in command_set:
            ratio = SequenceMatcher(None, text_lower, cmd.lower()).ratio()
            if ratio >= 0.5:
                print(f"命令模糊匹配成功: {text} ≈ {cmd} (相似度: {ratio:.2f})")
                return True
        
        return False

    # ====== [ASR模块-核心识别方法] 换ASR模型时替换整个方法的推理+解码逻辑 ======
    def recognize_with_onnx(self, audio_float):
        """
        ASR核心流程：输入float32音频数组 → 输出(text文本, 耗时秒数)
        
        Returns:
            (text: str, cost_sec: float)  — 文本内容 + 从送入到出结果的耗时
        """
        start = time.time()

        # Step 1: 提取MFCC特征 (T, 40)
        features = extract_mfcc_from_array(audio_float)

        if features.shape[0] == 0:
            return "", 0.0

        # Step 2: CMVN归一化
        features = cmvn(features)

        # Step 3: 添加batch维度 (1, T, 40)，适配ONNX输入格式
        features = np.expand_dims(features, axis=0)

        # Step 4: ONNX模型前向推理，输出logits (1, T, vocab_size)
        logits = self.model(features)

        # Step 5: CTC解码（去除重复和空白符）
        logits = logits.squeeze(axis=1)  # (1, T, vocab) -> (T, vocab)
        ids = logits.argmax(axis=-1)      # 取每帧概率最大的token ID

        tokens = []
        blank = 0       # CTC空白符ID
        prev = -1       # 前一个token ID（去重用）
        for k in ids:
            if k != blank and k != prev:
                tokens.append(k)
            prev = k

        # Step 6: token ID -> 中文字符 -> 文本拼接
        tokens = [self.id2token[i] for i in tokens]
        text = "".join(tokens)

        # 计时
        cost = time.time() - start
        
        # 区分首次预热/正常耗时标记
        if self.asr_first_run:
            print(f"📊 [ASR] 首次识别耗时（含CPU预热）: {cost:.3f} 秒")
            self.asr_first_run = False
        else:
            pass  # 正常耗时由调用方和文字一起打印，这里不单独打

        return text, cost
    # ====== [ASR模块-核心识别方法结束] ===================================

    def process_audio_buffer(self, audio_buffer, silent=False):
        """
        音频缓冲区处理（换ASR模型时一般不需改，只需改上面调用的recognize_with_onnx）
        
        Args:
            audio_buffer: 音频数据缓冲（bytes列表，每个元素是int16帧）
            silent: True时跳过"处理音频..."打印（唤醒后使用）
        """
        if len(audio_buffer) < 15:  # 至少 450ms 的音频
            return None

        # 合并音频 bytes -> int16 -> float32
        audio_data = b''.join(audio_buffer)
        audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 使用 ASR 模型进行转录
        if not silent:
            print(f"处理音频... ({len(audio_float) / SAMPLING_RATE:.1f}s)")
        
        # 根据模式选择ASR引擎
        if self.english_mode:
            # ----- 英文模式: 使用 fmodel Whisper -----
            text, cost = self._recognize_with_whisper_wrapper(audio_data)
            engine_tag = "[EN]"
        else:
            # ----- 中文模式: 使用 ONNX TeleSpeech -----
            text, cost = self.recognize_with_onnx(audio_float)
            engine_tag = "[CN]"

        if not silent and text:
            print(f"识别结果: {text}  ({engine_tag} ASR耗时: {cost:.3f}s)")
        return text, cost

    def _recognize_with_whisper_wrapper(self, audio_data_int16_bytes):
        """fmodel Whisper 包装：返回 (text, cost) 元组，与 ONNX 接口一致"""
        start = time.time()
        audio_np = np.frombuffer(audio_data_int16_bytes, dtype=np.int16)
        text = self._whisper_transcribe(audio_np)
        cost = time.time() - start
        return text, cost

    def _should_process_buffer(self):
        """判断是否应该立即处理当前缓冲区"""
        if len(self.audio_buffer) == 0:
            return False
        # 尾静默超限：连续静音帧数超过阈值，说明话已说完
        if self.trailing_silent_frames >= self.TRAILING_SILENT_MAX:
            return True
        # 语音过长：缓冲帧数超过上限（且已达到最小有效长度），强制处理
        if len(self.audio_buffer) >= self.MAX_SPEECH_FRAMES and len(self.audio_buffer) >= self.MIN_SPEECH_FOR_FLUSH:
            return True
        return False

    def _flush_and_recognize(self, silent_mode=False):
        """清空缓冲区并进行ASR识别+唤醒词匹配+命令检测（公共逻辑抽取）"""
        global _asr_result_callback, _wake_callback
        
        # 【计时】记录用户说完话的时间（VAD判断静音触发处理的时刻）
        user_speech_end_time = time.perf_counter()
        
        # ONE_SHOT_ASR模式下，已完成一次转录且已唤醒状态时，跳过
        # 但如果未唤醒（one_shot_done=True但is_awake=False），仍需检测唤醒词
        if ONE_SHOT_ASR and self.one_shot_done and self.is_awake:
            self.audio_buffer = []
            self.trailing_silent_frames = 0
            return

        result = self.process_audio_buffer(self.audio_buffer, silent=silent_mode)
        
        # 保存当前音频数据（智能情绪模式需要用于SER推理）
        _current_audio_float = None
        if self.emotion_enabled and self.smart_emotion_mode:
            audio_data = b''.join(self.audio_buffer)
            _current_audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        self.audio_buffer = []
        self.trailing_silent_frames = 0
        
        # 解包 (text, cost) 元组
        if result is None:
            return
        recognized_text, asr_cost = result
        
        if recognized_text:
            if not self.is_awake:
                # === 未唤醒状态：检测唤醒词 ===
                if self.is_fuzzy_match(recognized_text):
                    print(f"\n✓✓✓ 检测到唤醒词: {recognized_text}  ([ASR] {asr_cost:.3f}s)")
                    self.is_awake = True
                    self.one_shot_done = False
                    print("\n" + "="*50)
                    print("🎉 唤醒成功！现在可以开始对话...")
                    print("="*50)
                    
                    # 提示当前情绪模式状态
                    if self.smart_emotion_mode:
                        print("   [智能情绪模式 ON]")
                    else:
                        print("   💬 当前: 文本情绪识别模式（说\"打开智能情绪识别\"可升级）")
                    
                    # 提示当前语言模式
                    if self.english_mode:
                        print("   [EN 英文识别模式] （说 \"back to chinese\" 可切回中文）")
                    else:
                        print("   [CN 中文识别模式] （说\"切换英文\"可切到英文）")
                    
                    # 调用唤醒回调（link_test.py 设置的）
                    if self.wake_callback:
                        self.wake_callback()
                    if _wake_callback:
                        _wake_callback()
            else:
                # === 已唤醒状态：识别用户语音 ===
                print(f"【ASR转录】: {recognized_text}  ([ASR] {asr_cost:.3f}s)")
                
                # ---- 命令检测：检查是否触发了功能切换 ----
                cmd_detected = False
                
                # 智能情绪模式切换
                if self.emotion_enabled and not self.smart_emotion_mode:
                    if self.is_command_match(recognized_text, SMART_EMOTION_COMMANDS):
                        cmd_detected = True
                        print("\n" + "=" * 40)
                        print("🔑 检测到命令: 打开智能情绪识别")
                        
                        # 检查模型是否已预加载
                        _model_ready = (self.emotion_detector._ser_loaded if self.emotion_detector else False)
                        if not _model_ready:
                            print("   正在加载语音情绪模型...")
                        if self._load_ser_model_for_smart():
                            self.smart_emotion_mode = True
                            if _model_ready:
                                print(f"   ✅ 智能情绪识别已激活！（模型已预加载，零等待）")
                            else:
                                print(f"   ✅ 智能情绪识别已激活！（文本关键词 + 语音语气融合）")
                        else:
                            print(f"   ⚠️ 模型加载失败，继续使用文本模式")
                        print("=" * 40 + "\n")
                
                # 排除LLM模式切换相关文本，避免"切换到本地模型"等误触语言切换
                _is_llm_mode_switch = any(kw in recognized_text for kw in ("本地", "云端", "模型", "离线", "在线"))

                # 语言切换命令：中文模式 -> 英文
                if not _is_llm_mode_switch and self.is_command_match(recognized_text, ENGLISH_MODE_COMMANDS):
                    cmd_detected = True
                    if not self.english_mode:
                        print("\n" + "=" * 40)
                        print("🔤 检测到命令: 切换英文")

                        # 检查模型是否已预加载
                        _whisper_ready = self.whisper_loaded
                        if not _whisper_ready:
                            print("   正在加载 Whisper 模型...")
                        if self._load_whisper_model():
                            self.english_mode = True
                            if _whisper_ready:
                                print(f"   ✅ 已切换到英文识别模式（fmodel已预加载，零等待）")
                            else:
                                print(f"   ✅ 已切换到英文识别模式（fmodel Whisper tiny）")
                        else:
                            print(f"   ⚠️ Whisper 加载失败，继续中文模式")
                        print("=" * 40 + "\n")
                    else:
                        pass
                # 语言回切：英文模式 -> 中文（英文命令词）
                elif self.english_mode and self.is_command_match(recognized_text, BACK_TO_CHINESE_EN_COMMANDS):
                    cmd_detected = True
                    print("\n" + "=" * 40)
                    print("🔤 检测到命令: Back to Chinese")
                    self.english_mode = False
                    print(f"   ✅ 已切回中文识别模式（ONNX）")
                    print("=" * 40 + "\n")
                # 语言回切：中文模式下的中文命令（兜底）
                elif not _is_llm_mode_switch and self.is_command_match(recognized_text, CHINESE_MODE_COMMANDS):
                    cmd_detected = True
                    if self.english_mode:
                        print("\n" + "=" * 40)
                        print("🔤 检测到命令: 切换中文")
                        self.english_mode = False
                        print(f"   ✅ 已切回中文识别模式（ONNX）")
                        print("=" * 40 + "\n")

                # ====== [情绪模块-调用] ASR转录后自动进行情绪识别 ======
                if self.emotion_enabled and not cmd_detected:
                    self._run_emotion_analysis(recognized_text, _current_audio_float)
                # ====== [情绪模块-调用结束] ===================================

                # 【关键】调用外部回调函数，将用户语音文本和时间戳传递给 link_test.py
                if _asr_result_callback:
                    try:
                        _asr_result_callback(recognized_text, user_speech_end_time, asr_cost)
                    except Exception as e:
                        print(f"❌ ASR回调执行错误: {e}")

                # ONE_SHOT_ASR模式下，每次对话结束后都回到等待唤醒状态（无论中英文）
                if ONE_SHOT_ASR and not cmd_detected:
                    self.one_shot_done = True
                    self.is_awake = False
                    print("\n⏸️ ASR已暂停，等待再次唤醒...")

    # ====== [情绪模块-分析调用] 与 emo+en_pc.py 完全一致 ======
    def _run_emotion_analysis(self, recognized_text, audio_float=None):
        """
        对ASR识别文本执行情绪分析并输出结果（含分项耗时统计）
        
        输出示例：
          【ASR转录】: 你好  (ASR耗时: 0.051s)
             🎭 情绪识别: 😊 [开心] ID:3 置信度:0.85 [语音SER] (情绪耗时: 0.320s)
        """
        if not self.emotion_detector or not recognized_text:
            return
        
        emo_start = time.time()
        emotion_result = self.emotion_detector.detect(
            recognized_text,
            audio_float=audio_float,
            smart_mode=self.smart_emotion_mode
        )
        emo_cost = time.time() - emo_start
        
        self.last_emotion_result = emotion_result
        
        if emotion_result:
            fmt = self.emotion_detector.format_result(emotion_result)
            print(f"   🎭 情绪识别: {fmt}  ([情绪] {emo_cost:.3f}s)")
            
            if emotion_result['emotion'] in ('fearful', 'surprised'):
                print(f"   ⚠️  检测到{emotion_result['emoji']}【{emotion_result['label_cn']}】意图！关键词: {emotion_result.get('matched_keywords', [])}")
        
        if self._emotion_callback and emotion_result:
            try:
                self._emotion_callback(emotion_result, recognized_text)
            except Exception as e:
                print(f"   [情绪回调异常] {e}")
    
    def _load_ser_model_for_smart(self):
        """为智能模式加载SER模型（封装调用）"""
        if self.emotion_detector and hasattr(self.emotion_detector, '_load_ser_model'):
            return self.emotion_detector._load_ser_model()
        return False
    
    def set_emotion_callback(self, callback_func):
        """设置情绪识别结果回调函数"""
        self._emotion_callback = callback_func
    
    def get_last_emotion(self):
        """获取最近一次的情绪识别结果（供外部查询）"""
        return self.last_emotion_result
    
    def toggle_smart_mode(self, enabled=None):
        """
        切换智能情绪模式
        Args:
            enabled: True=开启, False=关闭, None=切换当前状态
        Returns:
            bool: 切换后的状态
        """
        if enabled is not None:
            self.smart_emotion_mode = enabled
        else:
            self.smart_emotion_mode = not self.smart_emotion_mode
        
        status = "ON ✅" if self.smart_emotion_mode else "OFF"
        print(f"   智能情绪模式: [{status}]")
        return self.smart_emotion_mode
    # ====== [情绪模块-分析调用结束] ===================================

    def pause(self):
        """暂停检测（LLM处理期间调用，减少终端输出和CPU占用）"""
        self._paused = True

    def resume(self):
        """恢复检测（LLM处理完毕后调用）"""
        self._paused = False

    def process_frame(self, data):
        """
        处理VAD音频帧（供外部调用）
        - 如果未唤醒，检测唤醒词
        - 如果已唤醒，识别并打印转录结果（ONE_SHOT_ASR模式下只转录一次）

        【优化】双重截断机制：
          1. 尾静默截断：连续静音帧≥阈值(~210ms)立即处理
          2. 语音超长兜底：缓冲帧数超上限(~6s)且达到最小有效长度时强制处理
             （防止VAD一直误判为"语音"导致永远不触发）
        """
        if self._paused:
            return self.is_awake

        if self.is_speech_detected(data):
            # 检测到语音 → 加入缓冲，重置静音计数
            self.audio_buffer.append(data)
            self.trailing_silent_frames = 0
            
            # 【新增】语音帧也要检查超限兜底！
            # 场景：VAD一直判语音（环境噪声/回声），尾静默永远到不了阈值
            #       但语音已经说了很久了，应该强制处理
            if self._should_process_buffer():
                silent_mode = self.is_awake
                self._flush_and_recognize(silent_mode)
        else:
            # 检测到静音
            if not self.audio_buffer:
                return self.is_awake
            
            # 静音帧计数 +1
            self.trailing_silent_frames += 1
            
            # 满足条件立即处理（不等VAD持续判静音）
            if self._should_process_buffer():
                silent_mode = self.is_awake
                self._flush_and_recognize(silent_mode)
        
        return self.is_awake


class IntegratedSystem:
    """
    整合系统：统一音频流，同时进行异响检测和VAD唤醒词检测
    
    【开发板适配】使用 arecord 子进程替代 PyAudio 录音，
    通过管道读取原始PCM数据，后续处理逻辑（VAD/异响/ASR）完全不变
    """
    
    def __init__(self, use_global_detector=True):
        # 初始化异响检测器
        self.monitor = SmartMonitor()
        
        # 初始化语音唤醒检测器（可选择使用全局单例）
        if use_global_detector and _global_detector is not None:
            self.wake_detector = _global_detector
        else:
            self.wake_detector = VoiceWakeDetector()
        
        self.wake_detector.wake_callback = self.on_wake
        
        # arecord 子进程（替代 PyAudio）
        self.arecord_proc = None
        
        # 标志位
        self.running = False
        
    def on_wake(self):
        """唤醒成功回调"""
        pass
    
    def audio_int16_to_float32(self, data):
        """将int16音频数据转换为float32（与原PyAudio版本完全一致）"""
        audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_float

    def _start_arecord(self):
        """
        启动 arecord 子进程进行持续录音（替代 PyAudio）
        
        输出: stdout 管道输出原始 PCM 数据 (S16_LE, 单声道, 16kHz)
              每帧 CHUNK_SIZE(480) bytes = 30ms 音频
        """
        cmd = [
            'arecord',
            '-D', ARECORD_DEVICE,       # 录音设备
            '-r', str(SAMPLING_RATE),     # 采样率 16000Hz
            '-f', ARECORD_FORMAT,         # 格式 S16_LE (与PyAudio paInt16一致)
            '-c', str(ARECORD_CHANNELS),  # 单声道
            '-t', 'raw',                  # 输出原始PCM（无wav头）
        ]
        self.arecord_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,       # PCM数据通过stdout管道传出
            stderr=subprocess.PIPE,
        )
        return self.arecord_proc

    def start(self):
        """启动整合系统 - 统一音频流"""
        self.running = True
        
        print("\n" + "="*55)
        print("🚀 启动异响检测 + 语音唤醒整合系统 [开发板版]")
        print("="*55)
        print(f"   录音设备: {ARECORD_DEVICE}")
        print(f"   采样率: {SAMPLING_RATE}Hz | 帧大小: {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLING_RATE*1000:.0f}ms)")
        print("\n📋 功能说明：")
        print("  • 统一音频流：arecord持续录音（子进程+管道）")
        print("  • 异响检测：每帧立即检测，不依赖VAD")
        print("  • 唤醒前：检测唤醒词'小光小光'等")
        print("  • 唤醒后：打印ASR转录结果")
        if EMOTION_ENABLED:
            print(f"  • 🎭 情绪识别：已启用（文本关键词模式）")
            print(f"     说「打开智能情绪识别」可升级为 语音+文本 融合模式")
        print(f"  • 🌐 语言切换：默认中文(ONNX)，唤醒后说「切换英文」切到fmodel Whisper，英文模式下说「back to chinese」回切")
        _whisper_label = '模型预加载(切换零等待)' if WHISPER_PRELOAD == '2' else '延迟加载(启动快,切换时加载)'
        print(f"     Whisper: {_whisper_label}  (修改 WHISPER_PRELOAD 变量可调整)")
        print(f"\n  • 🧠 ASR: ONNX TeleSpeech (中文) / fiboaisdk Whisper tiny (英文)")
        print("\n🎤 等待唤醒词... (Ctrl+C 停止)")
        print("-"*55)
        
        # 启动 arecord 子进程（替代 pyaudio.open）
        try:
            self._start_arecord()
        except FileNotFoundError:
            print(f"\n❌ 找不到 arecord 命令！请确认开发板已安装 alsa-utils")
            print("   安装命令: apt-get install alsa-utils")
            self.running = False
            return
        except Exception as e:
            print(f"\n❌ 启动 arecord 失败: {e}")
            self.running = False
            return
        
        print(f"\n🎤 arecord 已启动(PID={self.arecord_proc.pid})，开始监听...")
        
        # 每帧的字节数 (480 samples × 2 bytes/sample = 960 bytes)
        BYTES_PER_FRAME = CHUNK_SIZE * 2
        
        try:
            while self.running:
                # 【关键】用 select 实现非阻塞读取（避免卡死）
                # 如果 arecord 进程退出或出错，及时退出循环
                readable, _, err = select.select(
                    [self.arecord_proc.stdout], [], 
                    [self.arecord_proc.stdout], 
                    1.0  # 超时1秒（用于定期检查 running 标志）
                )
                
                if err:
                    print("\n❌ arecord 管道异常退出")
                    break
                
                if not readable:
                    continue  # 无数据可读，继续等待
                
                # 1. 从管道读取一帧原始 PCM 数据 (int16, 与PyAudio格式一致)
                data = self.arecord_proc.stdout.read(BYTES_PER_FRAME)
                
                if len(data) < BYTES_PER_FRAME:
                    # 数据不足说明 arecord 可能已结束或异常
                    if not self.running:
                        break
                    continue
                
                # 2. 转换为float32用于异响检测（与原代码完全一致）
                audio_float = self.audio_int16_to_float32(data)
                
                # 3. 【关键】立即进行异响检测（100%覆盖，不被VAD过滤）
                # 唤醒后silent=True，停止打印调试信息，只保留报警
                silent_mode = self.wake_detector.is_awake
                self.monitor.process_audio(audio_float, silent=silent_mode)
                
                # 4. 同时送入VAD进行唤醒词检测和ASR转录（与原代码完全一致）
                self.wake_detector.process_frame(data)
                
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号...")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止系统（终止arecord子进程）"""
        self.running = False
        if self.arecord_proc:
            try:
                self.arecord_proc.terminate()
                self.arecord_proc.wait(timeout=2)
            except Exception:
                self.arecord_proc.kill()
            self.arecord_proc = None
        print("🛑 系统已停止")


# ==================== 批量测试函数（来自integrate_beta copy.py）====================
def test_all_files(folder_path):
    """批量测试异响文件"""
    if not os.path.exists(folder_path): 
        print("路径不存在！")
        return

    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.wav')])
    if not files: 
        print("文件夹内没有WAV文件。")

    print(f"📂 正在分析 {len(files)} 个文件（仅显示异常报警）...\n")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        monitor = SmartMonitor()
        
        try:
            with wave.open(file_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if wf.getnchannels() == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                # 获取特征
                rms, centroid, high_freq_energy, _ = monitor.calculate_features(audio_data)
                
                # 与原文件 integrate_beta copy.py 批量测试一致的判断逻辑
                # 低频巨响：音量超过基线 + 高频能 > 阈值
                is_impact = (rms > monitor.baseline_vol * 1.5 + 0.02) and (high_freq_energy > HIGH_FREQ_ENERGY_THRESHOLD)
                # 高频异响：质心 > 阈值 + 高频能 > 阈值 + 音量 > 最小阈值
                is_glass = (centroid > FREQUENCY_THRESHOLD) and (high_freq_energy > HIGH_FREQ_ENERGY_THRESHOLD) and (rms > MIN_VOL_ABS)
                
                if is_impact or is_glass:
                    alarm_type = "🔨 低频巨响" if is_impact else "🔪 高频异响"
                    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    print(f"🚨 触发报警: {filename}")
                    print(f"   ⏰ 检测时间: {current_time_str}")
                    print(f"   🔍 报警类型: {alarm_type}")
                    print(f"   🔊 平均音量: {rms:.3f}")
                    print(f"   📊 频谱质心: {centroid:.0f}Hz")
                    print(f"   📈 高频能量: {high_freq_energy:.2f}")
                    print("-" * 30)
                
        except Exception as e:
            print(f"解析文件 {filename} 时出错: {e}")

    print("\n✅ 测试分析完成！")


# ==================== 外部衔接接口（供开发板上其他py文件调用）====================
# 与 备份/asr.py 保持完全一致的接口 + 新增情绪相关接口

# 全局ASR结果回调函数（由外部文件如link.py设置）
# 调用方式: from asr import set_asr_callback; set_asr_callback(your_function)
_asr_result_callback = None

# 全局唤醒回调函数
_wake_callback = None

# 全局 VoiceWakeDetector 单例（用于外部访问）
_global_detector = None


def init_asr():
    """
    【兼容接口】预加载ASR模型（全局单例模式）
    供 link_test.py 等外部文件调用，避免首次识别时的模型加载延迟
    """
    global _global_detector
    if _global_detector is None:
        print("🔄 正在预加载 ASR 模型...")
        _global_detector = VoiceWakeDetector()
        print("✅ ASR 模型预加载完成")
    return _global_detector


def set_asr_callback(callback_func):
    """
    设置ASR结果回调函数（开发板衔接用）
    
    Args:
        callback_func: 回调函数，签名为 func(text: str, user_speech_end_time: float, asr_cost: float) -> None
                       - text: ASR识别出的文本
                       - user_speech_end_time: 用户说完话的时间戳 (time.perf_counter())
                       - asr_cost: ASR推理耗时（秒）
                       唤醒后每次ASR识别出用户语音文本后会自动调用此函数
                       
    使用示例（在 link_test.py 中）:
        from asr import set_asr_callback
        def handle_asr_text(text, user_speech_end_time, asr_cost):
            print(f"收到ASR结果: {text}, 用户说完于: {user_speech_end_time:.3f}")
            # ... 调用LLM + TTS ...
        set_asr_callback(handle_asr_text)
    """
    global _asr_result_callback
    _asr_result_callback = callback_func
    print(f"✅ ASR回调函数已设置: {callback_func.__name__ if callback_func else None}")


def set_wake_callback(callback_func):
    """
    设置唤醒成功回调函数

    Args:
        callback_func: 回调函数，签名为 func() -> None
                       检测到唤醒词后会自动调用此函数
    """
    global _wake_callback
    _wake_callback = callback_func
    print(f"✅ 唤醒回调函数已设置: {callback_func.__name__ if callback_func else None}")


def pause_detection():
    """暂停ASR检测（LLM处理期间调用，减少终端输出和CPU占用）"""
    global _system_instance
    if _system_instance and _system_instance.wake_detector:
        _system_instance.wake_detector.pause()
        print("⏸️ ASR检测已暂停")


def resume_detection():
    """恢复ASR检测（LLM处理完毕后调用）"""
    global _system_instance
    if _system_instance and _system_instance.wake_detector:
        _system_instance.wake_detector.resume()
        print("▶️ ASR检测已恢复")


def transcribe_wav(wav_path):
    """
    【标准对外接口】转录wav文件（与asr (copy).py的transcribe_wav完全兼容）
    
    供 cloud_method/link.py 等外部文件调用。
    内部自动使用已加载的ONNX模型进行识别，无需重复初始化。
    
    Args:
        wav_path: wav文件绝对路径
        
    Returns:
        str: 识别出的文本字符串
        
    使用示例:
        from asr import transcribe_wav
        result = transcribe_wav("/path/to/audio.wav")
    """
    if not wav_path:
        raise ValueError("wav path is empty")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"wav not found: {wav_path}")

    # 读取音频文件 → float32数组
    data, sr = sf.read(wav_path, dtype="float32")
    if len(data.shape) > 1:
        data = data[:, 0]  # 取单声道
    if sr != SAMPLING_RATE:
        # 重采样到16kHz（需要librosa，如果没有则跳过并警告）
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLING_RATE)
        except ImportError:
            print(f"[WARNING] 采样率{sr} != {SAMPLING_RATE}Hz，且未安装librosa，可能导致识别不准")

    # 复用 VoiceWakeDetector 的 ONNX 推理流程
    detector = VoiceWakeDetector()
    text, cost = detector.recognize_with_onnx(data.astype(np.float32))
    
    result_text = str((text or "").strip())
    print(f"📊 transcribe_wav 结果: '{result_text}'  ({cost:.3f}s)")
    
    return result_text


def get_alarm_status():
    """
    查询接口：获取当前异响检测状态（供外部文件轮询）
    
    Returns:
        dict: 包含系统运行状态、最近报警时间、唤醒状态、音量基线等
    """
    # 全局系统实例引用（需要在 IntegratedSystem 启动后才能使用）
    global _system_instance
    if _system_instance is None:
        return {'running': False, 'last_alarm_time': 0, 'is_awake': False, 'baseline_vol': 0}
    
    return {
        'running': _system_instance.running,
        'last_alarm_time': _system_instance.monitor.last_alarm_time,
        'is_awake': _system_instance.wake_detector.is_awake,
        'baseline_vol': _system_instance.monitor.baseline_vol,
    }


# ==================== 情绪识别外部接口（供其他py文件调用，与 emo+en_pc.py 一致）====================
def analyze_emotion(text):
    """
    【标准对外接口】分析文本情绪（独立调用，无需启动完整系统）
    
    Args:
        text: 待分析的文本字符串
        
    Returns:
        dict: 情绪识别结果 或 None
        
    使用示例:
        from asr import analyze_emotion
        result = analyze_emotion("我今天好开心啊")
        print(result['label'])  # → "积极"
    """
    if not text or not text.strip():
        return None
    detector = EmotionDetector(ser_model_path=EMOTION_SER_MODEL_PATH if EMOTION_SER_MODEL_PATH else None,
                              rules_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_rules.json"))
    return detector.detect(text)


def set_emotion_callback(callback_func):
    """
    设置情绪结果回调函数
    
    使用示例:
        from asr import set_emotion_callback
        def handle_emotion(emotion_result, asr_text):
            print(f"情绪: {emotion_result['label']}, 文本: {asr_text}")
        set_emotion_callback(handle_emotion)
    """
    global _system_instance
    if _system_instance and hasattr(_system_instance, 'wake_detector'):
        _system_instance.wake_detector.set_emotion_callback(callback_func)


def get_last_emotion():
    """
    查询接口：获取最近一次的情绪识别结果
    Returns: dict or None
    """
    global _system_instance
    if _system_instance and hasattr(_system_instance, 'wake_detector'):
        return _system_instance.wake_detector.get_last_emotion()
    return None


# 全局系统实例（用于外部查询接口）
_system_instance = None

# ===================== 主程序 =====================
if __name__ == "__main__":
    AUDIO_FOLDER = r"/home/fibo/AI model/llm_models/abnormal_test"
    
    print("🤖 异响检测 + 语音唤醒 + 情绪识别 + 英文模式 整合系统 [开发板版]")
    _preload_label = '模型预加载(说命令零等待)' if SMART_EMOTION_PRELOAD == '2' else '延迟加载(启动快,说命令后加载)'
    print(f"   情绪模型: {_preload_label}")
    _whisper_label = '模型预加载(切换零等待)' if WHISPER_PRELOAD == '2' else '延迟加载(启动快,切换时加载)'
    print(f"   Whisper: {_whisper_label}")
    print("1 = 批量测试 (仅显示报警)")
    print("2 = 实时监听 + 唤醒（统一音频流）")
    choice = "2"  # 默认模式2，实时监听 + 唤醒

    if choice == "1":
        test_all_files(AUDIO_FOLDER)
    elif choice == "2":
        system = IntegratedSystem()
        _system_instance = system  # 注册全局实例，供外部查询接口使用
        system.start()
