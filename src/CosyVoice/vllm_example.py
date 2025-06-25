import sys
import logging
import os
from typing import Optional
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
import asyncio
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)


import io
import torch
import traceback
import torchaudio
import numpy as np
import requests
import tempfile
from typing import List
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from fastapi import WebSocket, WebSocketDisconnect
from torchaudio.io import CodecConfig
from pydantic import ValidationError


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
    
    # 音频编码相关参数
    bit_rate: Optional[int] = 192000        # 比特率，默认 192kbps
    compression_level: Optional[int] = 2     # 压缩级别 (0-9)
    

    @field_validator('speed')
    @classmethod
    def check_speed(cls, v):
        if v <= 0:
            raise ValueError('Speed must be positive')
        return v

    @field_validator('mode')
    @classmethod
    def check_mode(cls, v):
        if v not in ["zero_shot", "cross_lingual", "instruct2"]:
            raise ValueError('Invalid mode')
        return v
    
    @field_validator('output_format')
    @classmethod
    def check_format(cls, v):
        if v not in ["pcm", "wav", "mp3"]:
            raise ValueError('Unsupported audio format')
        return v

    @field_validator('bit_rate')
    @classmethod
    def check_bit_rate(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Bit rate must be positive')
        return v
    
    @field_validator('compression_level')
    @classmethod
    def check_compression_level(cls, v):
        if v is not None and (v < 0 or v > 9):
            raise ValueError('Compression level must be between 0 and 9')
        return v
    
    @field_validator('channels')
    @classmethod
    def check_channels(cls, v):
        if v not in [1, 2]:
            raise ValueError('Channels must be either 1 or 2')
        return v

def convert_to_format(audio_tensor, format, sample_rate, request):
    """转换音频格式"""
    if format == "pcm":
        if request.channels == 2 and audio_tensor.size(0) == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        audio_array = (audio_tensor * 32767).numpy().astype(np.int16)
        return audio_array.tobytes()
    else:
        # WAV或MP3格式
        buffer = io.BytesIO()
        if request.channels == 2 and audio_tensor.size(0) == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        
        if format == "mp3":
            config = CodecConfig(
                bit_rate=request.bit_rate or 192000,
                compression_level=request.compression_level or 2,
            )
            torchaudio.save(buffer, audio_tensor, sample_rate, format='mp3', compression=config)
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
                        request
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
                    request
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
   
#  Websocket Manager
class ConnectionManager:
    def __init__(self, max_connections=100):
        self.active_connections: List[WebSocket] = []
        self.max_connections = max_connections

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Max connections reached")
            raise WebSocketDisconnect("Max connections reached")
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_audio(self, audio_data: bytes, websocket: WebSocket):
        try:
            await websocket.send_bytes(audio_data)
        except Exception as e:
            logger.error(f"Error sending audio data: {e}")
            raise

    async def heartbeat(self, websocket: WebSocket):
        """心跳检测"""
        try:
            while True:
                await websocket.send_json({"type": "ping"})
                await asyncio.sleep(30)  # 每30秒发送一次心跳
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            await self.disconnect(websocket)

manager = ConnectionManager(max_connections=100)

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    # 启动心跳任务
    heartbeat_task = None
    try:
        await manager.connect(websocket)
        # 启动心跳检测
        heartbeat_task = asyncio.create_task(manager.heartbeat(websocket))
        
        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict) and data.get("type") == "pong":
                logger.debug("Received pong message")
                continue
            
            try:
                request = TTSRequest(**data)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid request format"
                })
                continue
                
            logger.info(f"Received TTS request for text: {request.text[:50]}...")
            
            temp_file = None
            try:
                # 处理音频文件路径
                if request.prompt_audio_path.startswith(('http://', 'https://')):
                    logger.info(f"Downloading audio from URL: {request.prompt_audio_path}")
                    temp_file = download_audio_file(request.prompt_audio_path)
                    prompt_speech_16k = load_wav(temp_file, 16000)
                    logger.info("Successfully loaded prompt audio from URL")
                else:
                    if not os.path.exists(request.prompt_audio_path):
                        await websocket.send_json({
                            "type": "error",
                            "message": "Prompt audio file not found"
                        })
                        continue
                    prompt_speech_16k = load_wav(request.prompt_audio_path, 16000)

                # 生成音频
                sample_rate = request.sample_rate or cosyvoice.sample_rate
                
                # 选择推理模式
                inference_generator = None
                if request.mode == "zero_shot":
                    inference_generator = cosyvoice.inference_zero_shot(
                        request.text, 
                        request.prompt_text, 
                        prompt_speech_16k, 
                        stream=True,
                        speed=request.speed
                    )
                elif request.mode == "cross_lingual":
                    inference_generator = cosyvoice.inference_cross_lingual(
                        request.text,
                        prompt_speech_16k,
                        stream=True,
                        speed=request.speed
                    )
                elif request.mode == "instruct2":
                    inference_generator = cosyvoice.inference_instruct2(
                        request.text,
                        request.instruct_text,
                        prompt_speech_16k,
                        stream=True,
                        speed=request.speed
                    )
                
                # 发送开始信号
                await websocket.send_json({
                    "type": "start",
                    "message": "Starting audio generation"
                })

                # 处理音频数据
                chunk_num = 0
                total_chunks = len(request.text) // 3  # 估算总块数
                
                for chunk in inference_generator:
                    chunk_num += 1
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
                        request
                    )
                    
                    # 发送音频数据
                    await manager.send_audio(audio_data, websocket)
                    
                    # 发送进度信息
                    progress = min(100, int((chunk_num / total_chunks) * 100))
                    await websocket.send_json({
                        "type": "progress",
                        "chunk": chunk_num,
                        "total": total_chunks,
                        "percentage": progress
                    })
                
                # 发送完成信号
                await websocket.send_json({
                    "type": "completed",
                    "total_chunks": chunk_num,
                    "message": "Audio generation completed"
                })

            except Exception as e:
                logger.error(f"Error in websocket_tts: {str(e)}")
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.info(f"Cleaned up temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)