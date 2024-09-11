from gpt_sovits import Factory
import os
import uuid
from io import BytesIO
import numpy as np
import soundfile as sf
from scipy.signal import resample
import subprocess
import wave


def normalize_audio(data: np.ndarray):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return data


def resample_audio(data: np.ndarray, original_rate: int, target_rate: int):
    data = normalize_audio(data)
    number_of_samples = round(len(data) * float(target_rate) / original_rate)
    resampled_data = resample(data, number_of_samples)
    resampled_data = normalize_audio(resampled_data)
    return resampled_data


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    """modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files"""
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, original_rate: int, target_rate: int, media_type: str):
    if target_rate and target_rate != original_rate:
        data = resample_audio(data, original_rate, target_rate)
        print("pack audio data processed")
        rate = target_rate
    else:
        rate = original_rate

    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_audio(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


cfg = Factory.read_config("/mnt/data/hanxiao/MyCode/Simiply-GPT-SoVITS/config/example.yaml")
model = Factory.build_model(cfg)

inputs = {
    "prompt_audio": "/mnt/data/hanxiao/MyCode/Simiply-GPT-SoVITS/example/jay_example.wav",
    "prompt_text": "你喜欢这样的东西，所以你我觉得人要有一技之长呢。",
    "text": "明月几时有,把酒问青天。"
}
sr, audio_data = model.generate(inputs)

audio_data = pack_audio(BytesIO(), audio_data, sr, 16000, 'wav').getvalue()

name = uuid.uuid4().hex
output_dir = os.getcwd()
output_file = os.path.join(output_dir, name + '.wav')
with wave.open(output_file, 'wb') as wf:
    # 设置声道数，采样宽度（字节），采样率和帧数
    wf.setnchannels(1)  # 假设是单声道
    wf.setsampwidth(2)  # 假设是16位深度，即2字节
    wf.setframerate(16000)  # 采样率
    wf.writeframesraw(audio_data[44:])  # 写入音频数据
