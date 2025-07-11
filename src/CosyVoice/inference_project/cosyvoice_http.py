import torch
from datetime import datetime
import traceback
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
import logging
import os
from typing import Optional
# from model import cosyvoice_model
import model
from utils import convert_to_format, download_audio_file, AudioProcessingError, get_media_type, cleanup_temp_file
from cosyvoice.utils.file_utils import load_wav

# 日志
logger = logging.getLogger(__name__)
router = APIRouter()

class TTSRequest(BaseModel):
    text: str
    mode: str = "zero_shot"
    prompt_audio_path: str = "./asset/zero_shot_prompt.wav"
    prompt_text: str = "希望你以后能够做的比我还好呦。"
    instruct_text: Optional[str] = ""
    speed: float = 1.0
    stream: bool = False
    output_format: str = "wav"  # pcm, wav, mp3
    zero_shot_spk_id: str = ''
    
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

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    temp_file = None
    try:
        # 处理音频文件路径
        if request.prompt_audio_path.startswith(('http://', 'https://')):
            try:
                logger.info(f"Downloading audio from URL: {request.prompt_audio_path}")
                temp_file = await download_audio_file(request.prompt_audio_path)
                prompt_speech_16k = load_wav(temp_file, 16000)
                logger.info("Successfully loaded prompt audio from URL")
            except Exception as e:
                raise AudioProcessingError(
                    status_code = 400,
                    detail = f"Error processing prompt audio file: {str(e)}"
                )
        else:
            if not os.path.exists(request.prompt_audio_path):
                raise AudioProcessingError(
                    message = "Prompt audio file not found",
                    status_code = 400,
                    details = {"path": request.prompt_audio_path}
                )
            prompt_speech_16k = load_wav(request.prompt_audio_path, 16000)

        async def generate_audio():
            audio_data = None
            chunk_num = 0

            try:
                if request.mode == "zero_shot":
                    inference_generator = model.cosyvoice_model.inference_zero_shot(
                        request.text, 
                        request.prompt_text, 
                        prompt_speech_16k, 
                        stream=request.stream,
                        speed=request.speed,
                        zero_shot_spk_id=request.zero_shot_spk_id
                    )
                elif request.mode == "cross_lingual":
                    inference_generator = model.cosyvoice_model.inference_cross_lingual(
                        request.text,
                        prompt_speech_16k,
                        stream=request.stream,
                        speed=request.speed,
                        zero_shot_spk_id=request.zero_shot_spk_id
                    )
                elif request.mode == "instruct2":
                    inference_generator = model.cosyvoice_model.inference_instruct2(
                        request.text,
                        request.instruct_text,
                        prompt_speech_16k,
                        stream=request.stream,
                        speed=request.speed,
                        zero_shot_spk_id=request.zero_shot_spk_id
                    )
                else:
                    raise ValueError(f"Unsupported mode: {request.mode}")
            
                # 收集所有音频数据
                for chunk in inference_generator:
                    chunk_num += 1
                    if request.stream:
                        audio_chunk = chunk['tts_speech']
                        
                        # 转换格式
                        audio_data = convert_to_format(
                            audio_chunk,
                            request.output_format,
                            model.cosyvoice_model.sample_rate,
                            request
                        )
                        
                        yield audio_data
                    else:
                        audio_data = torch.cat([audio_data, chunk['tts_speech']], dim=1) if audio_data is not None else chunk['tts_speech']

                if not request.stream and audio_data is not None:
                    
                    # 转换格式
                    final_audio = convert_to_format(
                        audio_data,
                        request.output_format,
                        model.cosyvoice_model.sample_rate,
                        request
                    )
                    yield final_audio

                logger.info(f"Task completed, generated {chunk_num} chunks")
            
            except Exception as e:
                logger.error(f"Error generating audio: {e}")
                raise AudioProcessingError(
                    message="Error generating audio",
                    status_code=500,
                    details={"error": str(e)}
                )

        media_type = get_media_type(request.output_format)
        
        headers = {
            "Content-Disposition": f"attachment; filename=output.{request.output_format}",
            "X-Processing-Time": str(datetime.now().timestamp())
        }

        if request.stream:
            return StreamingResponse(generate_audio(), media_type=media_type, headers=headers)
        else:
            audio_data = b''.join([chunk async for chunk in generate_audio()])
            return Response(content=audio_data, media_type=media_type, headers=headers)

    except AudioProcessingError as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "timestamp": datetime.now().isoformat()}
        )
    finally:
        # 清理临时文件
        if temp_file:
            cleanup_temp_file(temp_file)
            
@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "tts"
    }

@router.get("/info")
async def get_info():
    """获取服务信息接口"""
    return {
        "version": "1.0.0",
        "model": "CosyVoice2",
        "supported_modes": ["zero_shot", "cross_lingual", "instruct2"],
        "supported_formats": ["pcm", "wav", "mp3"],
        "sample_rate": model.cosyvoice_model.sample_rate,
        "streaming_supported": True
    }