import requests
import io
import pyaudio
import wave
import json

def stream_audio(url, json_data):
    try:
        response = requests.post(url, json=json_data, stream=True)
        response.raise_for_status()  # 如果响应状态不是200，将引发异常
        
        # 初始化 PyAudio
        p = pyaudio.PyAudio()
        first_chunk = True
        stream = None

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                # 读取音频格式信息（仅在第一个块时进行）
                if first_chunk:
                    wav_file = io.BytesIO(chunk)
                    with wave.open(wav_file, 'rb') as wav:
                        channels = wav.getnchannels()
                        sample_width = wav.getsampwidth()
                        framerate = wav.getframerate()
                    
                    # 打开音频流
                    stream = p.open(format=p.get_format_from_width(sample_width),
                                    channels=channels,
                                    rate=framerate,
                                    output=True)
                    first_chunk = False
                
                # 播放音频
                stream.write(chunk)

        # 关闭音频流和 PyAudio
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    except requests.RequestException as e:
        print(f"请求错误: {e}")
    except Exception as e:
        print(f"播放音频时发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    url = "http://localhost:8000/tts"
    data = {
        "text": "这是一个测试文本，用于演示文本到语音的转换。",
        "mode": "zero_shot",
        "prompt_audio_path": "./asset/zero_shot_prompt.wav",
        "prompt_text": "希望你以后能够做的比我还好呦。",
        "speed": 1.0,
        "stream": True
    }

    print("开始请求音频流...")
    stream_audio(url, data)
    print("音频播放完成。")