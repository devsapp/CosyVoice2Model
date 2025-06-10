import sys
import logging
import os
from typing import Optional
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import io
import torch
import traceback
import torchaudio
import numpy as np
import requests
import tempfile
from urllib.parse import urlparse

app = FastAPI()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 CosyVoice2 模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)

class TTSRequest(BaseModel):
    text: str
    mode: str = "zero_shot"
    prompt_audio_path: str = "./asset/zero_shot_prompt.wav"
    prompt_text: str = "希望你以后能够做的比我还好呦。"
    instruct_text: Optional[str] = ""
    speed: float = 1.0
    stream: bool = False
    output_format: str = "wav"  # pcm, wav, mp3
    audio_quality: Optional[int] = None  # MP3质量 (0-9)
    sample_rate: Optional[int] = None
    channels: int = 1

    @validator('speed')
    def check_speed(cls, v):
        if v <= 0:
            raise ValueError('Speed must be positive')
        return v

    @validator('mode')
    def check_mode(cls, v):
        if v not in ["zero_shot", "cross_lingual", "instruct2"]:
            raise ValueError('Invalid mode')
        return v
    
    @validator('output_format')
    def check_format(cls, v):
        if v not in ["pcm", "wav", "mp3"]:
            raise ValueError('Unsupported audio format')
        return v

    @validator('audio_quality')
    def check_quality(cls, v):
        if v is not None and (v < 0 or v > 9):
            raise ValueError('Audio quality must be between 0 and 9')
        return v

def convert_to_format(audio_tensor, format, sample_rate, channels, quality=None):
    """转换音频格式"""
    if format == "pcm":
        # 转换为PCM格式
        if channels == 2 and audio_tensor.size(0) == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        # 转换为16位整数PCM
        audio_array = (audio_tensor * 32767).numpy().astype(np.int16)
        return audio_array.tobytes()
    else:
        # WAV或MP3格式
        buffer = io.BytesIO()
        if channels == 2 and audio_tensor.size(0) == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        
        if format == "mp3" and quality is not None:
            torchaudio.save(buffer, audio_tensor, sample_rate, format='mp3', compression=quality)
        else:
            torchaudio.save(buffer, audio_tensor, sample_rate, format=format)
        
        return buffer.getvalue()

def download_audio_file(url: str, timeout: int = 30, max_size: int = 10 * 1024 * 1024):
    """下载音频文件并进行验证"""
    try:
        # 验证 URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL")

        # 下载文件，带进度和大小限制
        response = requests.get(
            url, 
            timeout=timeout,
            stream=True,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        response.raise_for_status()

        # 检查内容类型
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('audio/'):
            logger.warning(f"Unexpected content type: {content_type}, but continuing anyway")

        # 检查文件大小
        content_length = int(response.headers.get('Content-Length', 0))
        if content_length > max_size:
            raise ValueError(f"File too large: {content_length} bytes")

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            return temp_file.name

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error downloading audio file: {str(e)}"
        )

@app.post("/tts")
def text_to_speech(request: TTSRequest):
    temp_file = None
    try:
        # 处理音频文件路径
        if request.prompt_audio_path.startswith(('http://', 'https://')):
            try:
                logger.info(f"Downloading audio from URL: {request.prompt_audio_path}")
                temp_file = download_audio_file(request.prompt_audio_path)
                prompt_speech_16k = load_wav(temp_file, 16000)
                logger.info("Successfully loaded prompt audio from URL")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing prompt audio file: {str(e)}"
                )
        else:
            if not os.path.exists(request.prompt_audio_path):
                raise HTTPException(status_code=400, detail="Prompt audio file not found")
            prompt_speech_16k = load_wav(request.prompt_audio_path, 16000)

        def generate(is_stream):
            audio_data = None
            chunk_num = 0
            header_sent = False

            inference_generator = None
            if request.mode == "zero_shot":
                inference_generator = cosyvoice.inference_zero_shot(
                    request.text, 
                    request.prompt_text, 
                    prompt_speech_16k, 
                    stream=is_stream,
                    speed=request.speed
                )
            elif request.mode == "cross_lingual":
                inference_generator = cosyvoice.inference_cross_lingual(
                    request.text,
                    prompt_speech_16k,
                    stream=is_stream,
                    speed=request.speed
                )
            elif request.mode == "instruct2":
                inference_generator = cosyvoice.inference_instruct2(
                    request.text,
                    request.instruct_text,
                    prompt_speech_16k,
                    stream=is_stream,
                    speed=request.speed
                )
            else:
                raise ValueError(f"Unsupported mode: {request.mode}")

            sample_rate = request.sample_rate or cosyvoice.sample_rate
            
            # 收集所有音频数据
            for chunk in inference_generator:
                chunk_num += 1
                if is_stream:
                    audio_chunk = chunk['tts_speech']
                    
                    # 重采样（如果需要）
                    if sample_rate != cosyvoice.sample_rate:
                        resampler = torchaudio.transforms.Resample(
                            cosyvoice.sample_rate, sample_rate
                        )
                        audio_chunk = resampler(audio_chunk)
                    
                    # 转换格式
                    audio_data = convert_to_format(
                        audio_chunk,
                        request.output_format,
                        sample_rate,
                        request.channels,
                        request.audio_quality
                    )
                    
                    yield audio_data
                else:
                    audio_data = torch.cat([audio_data, chunk['tts_speech']], dim=1) if audio_data is not None else chunk['tts_speech']

            if not is_stream and audio_data is not None:
                # 重采样（如果需要）
                if sample_rate != cosyvoice.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        cosyvoice.sample_rate, sample_rate
                    )
                    audio_data = resampler(audio_data)
                
                # 转换格式
                final_audio = convert_to_format(
                    audio_data,
                    request.output_format,
                    sample_rate,
                    request.channels,
                    request.audio_quality
                )
                
                yield final_audio

            logger.info(f"Task completed, generated {chunk_num} chunks")

        # 设置正确的媒体类型
        media_types = {
            "pcm": "audio/l16",  # 线性16位PCM
            "wav": "audio/wav",
            "mp3": "audio/mpeg"
        }
        media_type = media_types[request.output_format]

        if request.stream:
            return StreamingResponse(generate(True), media_type=media_type)
        else:
            audio_data = b''.join(generate(False))
            return Response(content=audio_data, media_type=media_type)

    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)