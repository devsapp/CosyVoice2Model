import io
import torch
import torchaudio
import numpy as np
import aiohttp
import tempfile
import os
import logging
import asyncio
from enum import Enum
from fastapi import HTTPException
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from torchaudio.io import CodecConfig
from pydantic import BaseModel

# 配置日志
logger = logging.getLogger(__name__)

async def download_audio_file(url: str, timeout: int = 60, max_size: int = 10 * 1024 * 1024) -> str:
    """
    下载音频文件并进行验证
    
    Args:
        url: 音频文件URL
        timeout: 超时时间（秒）
        max_size: 最大文件大小（字节）
    
    Returns:
        临时文件路径
    """
    try:
        # 验证 URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise AudioProcessingError(
                message="Invalid URL format",
                details={"url": url}
            )

        # 下载文件
        timeout_client = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_client) as session:
            try:
                async with session.get(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                ) as response:
                    if response.status != 200:
                        raise AudioProcessingError(
                            message=f"Failed to download file: HTTP {response.status}",
                            details={
                                "url": url,
                                "status": response.status,
                                "reason": response.reason
                            }
                        )

                    # 检查内容类型
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('audio/'):
                        logger.warning(f"Unexpected content type: {content_type}, but continuing anyway")

                    # 检查文件大小
                    content_length = int(response.headers.get('Content-Length', 0))
                    if content_length > max_size:
                        raise AudioProcessingError(
                            message="File too large",
                            details={
                                "size": content_length,
                                "max_size": max_size
                            }
                        )

                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        total_size = 0
                        async for chunk in response.content.iter_chunked(8192):
                            total_size += len(chunk)
                            if total_size > max_size:
                                os.unlink(temp_file.name)
                                raise AudioProcessingError(
                                    message="File too large during download",
                                    details={
                                        "size": total_size,
                                        "max_size": max_size
                                    }
                                )
                            temp_file.write(chunk)
                        return temp_file.name

            except asyncio.TimeoutError:
                raise AudioProcessingError(
                    message="Download timeout",
                    details={
                        "url": url,
                        "timeout": timeout
                    }
                )
            except aiohttp.ClientError as e:
                raise AudioProcessingError(
                    message="Network error during download",
                    details={
                        "url": url,
                        "error": str(e)
                    }
                )

    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(
            message="Unexpected error during download",
            status_code=500,
            details={"error": str(e)}
        )

def convert_to_format(audio_tensor, format: str, sample_rate: int, request) -> bytes:
    """
    转换音频格式
    
    Args:
        audio_tensor: 音频张量
        format: 目标格式 (pcm/wav/mp3)
        sample_rate: 采样率
        request: 请求参数对象
    
    Returns:
        转换后的音频数据
    """
    if format == "pcm":
        if hasattr(request, 'channels') and request.channels == 2 and audio_tensor.size(0) == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        audio_array = (audio_tensor * 32767).numpy().astype(np.int16)
        return audio_array.tobytes()
    else:
        # WAV或MP3格式
        buffer = io.BytesIO()
        if hasattr(request, 'channels') and request.channels == 2 and audio_tensor.size(0) == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        
        try:
            if format == "mp3":
                config = CodecConfig(
                    bit_rate=getattr(request, 'bit_rate', 192000),
                    compression_level=getattr(request, 'compression_level', 2),
                )
                torchaudio.save(buffer, audio_tensor, sample_rate, format='mp3', compression=config)
            else:
                torchaudio.save(buffer, audio_tensor, sample_rate, format=format)
            
            return buffer.getvalue()
        finally:
            buffer.close()

def cleanup_temp_file(file_path: str):
    """
    清理临时文件
    
    Args:
        file_path: 文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to delete temporary file {file_path}: {e}")

def get_media_type(format: str) -> str:
    """
    获取媒体类型
    
    Args:
        format: 音频格式
    
    Returns:
        对应的媒体类型
    """
    media_types = {
        "pcm": "audio/l16",  # 线性16位PCM
        "wav": "audio/wav",
        "mp3": "audio/mpeg"
    }
    return media_types.get(format.lower(), "application/octet-stream")

class AudioProcessingError(Exception):
    """音频处理相关的自定义异常"""
    
    def __init__(self, message: str, status_code: int = 400, details: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self):
        return {
            "error": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": datetime.now().isoformat()
        }
        
# websocket 相关
# 错误码
class ErrorCode(str, Enum):
    """WebSocket错误代码枚举"""
    INVALID_PARAMS = "INVALID_PARAMS"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    INVALID_SESSION = "INVALID_SESSION"
    SYNTHESIS_ERROR = "SYNTHESIS_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    
class ResponseBuilder:
    """WebSocket响应构建器"""
    @staticmethod
    def create_response(message: dict, message_type: str, payload: dict, status: str = "success") -> dict:
        return {
            "header": {
                "version": "1.0",
                "message_type": message_type,
                "timestamp": datetime.now().isoformat(),
                "sequence": message["header"]["sequence"]
            },
            "payload": {
                "status": status,
                **payload
            }
        }

    @staticmethod
    def create_error_response(message: dict, error_code: ErrorCode, error_message: str, details: Optional[str] = None) -> dict:
        payload = {
            "status": "error",
            "error": error_message,
            "error_code": error_code
        }
        if details:
            payload["details"] = details

        return ResponseBuilder.create_response(
            message=message,
            message_type=f"{message['header']['message_type']}_RESPONSE",
            payload=payload
        )

def validate_message(message: dict, supported_versions: list) -> Tuple[bool, Optional[str]]:
    """验证WebSocket消息格式"""
    try:
        if message["header"]["version"] not in supported_versions:
            return False, "Unsupported version"
        
        required_fields = ["message_type", "sequence", "timestamp"]
        for field in required_fields:
            if field not in message["header"]:
                return False, f"Missing required field: {field}"

        return True, None
    except Exception as e:
        return False, str(e)