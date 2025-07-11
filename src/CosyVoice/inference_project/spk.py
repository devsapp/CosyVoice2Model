from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
import logging
import os
from typing import Optional
# from model import cosyvoice_model
import model
from utils import convert_to_format, download_audio_file, AudioProcessingError
from cosyvoice.utils.file_utils import load_wav

logger = logging.getLogger(__name__)
router = APIRouter()

class SpeakerRequest(BaseModel):
    prompt_text: str = Field(
        default="希望你以后能够做的比我还好呦。",
        description="Text prompt for speaker registration"
    )
    zero_shot_spk_id: str = Field(
        default="my_zero_shot_spk",
        description="Speaker ID to register"
    )
    prompt_audio_path: str = Field(
        default="./asset/zero_shot_prompt.wav",
        description="Path or URL to prompt audio file"
    )

@router.post("/spk", response_model=dict)
async def speaker_id(request: SpeakerRequest) -> dict:
    """Register a new speaker ID with provided audio and text prompt."""
    temp_file = None
    try:    
        # 音频文件处理
        if request.prompt_audio_path.startswith(('http://', 'https://')):
            try:
                logger.info(f"Downloading audio from URL: {request.prompt_audio_path}")
                temp_file = await download_audio_file(request.prompt_audio_path)
                prompt_speech_16k = load_wav(temp_file, 16000)
                logger.info("Successfully loaded prompt audio from URL")
            except Exception as e:
                logger.error(f"Error downloading audio: {str(e)}")
                raise AudioProcessingError(
                    status_code=400,
                    message="Error processing prompt audio file",
                    details={"error": str(e)}
                )
        else:
            if not os.path.exists(request.prompt_audio_path):
                logger.error(f"Audio file not found: {request.prompt_audio_path}")
                raise AudioProcessingError(
                    message="Prompt audio file not found",
                    status_code=400,
                    details={"path": request.prompt_audio_path}
                )
            try:
                prompt_speech_16k = load_wav(request.prompt_audio_path, 16000)
            except Exception as e:
                logger.error(f"Error loading audio: {str(e)}")
                raise AudioProcessingError(
                    message="Error loading audio file",
                    status_code=400,
                    details={"error": str(e)}
                )
        
        try:
            spk_success = model.cosyvoice_model.add_zero_shot_spk(
                request.prompt_text,
                prompt_speech_16k,
                request.zero_shot_spk_id
            )
            return {
                "spk_success": spk_success,
                    "spk_id": request.zero_shot_spk_id
                    }
        except Exception as e:
            logger.error(f"Error registering speaker: {str(e)}")
            return {
                "spk_success": False,
                    "spk_id": request.zero_shot_spk_id
                    }
    
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")