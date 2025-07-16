from fastapi import APIRouter, HTTPException, Response, Path
from pydantic import BaseModel, Field
import logging
import os
from typing import Optional, Dict, List
# from model import cosyvoice_model
import model
from utils import convert_to_format, download_audio_file, AudioProcessingError
from cosyvoice.utils.file_utils import load_wav
from datetime import datetime

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
    
class SpeakerCreate(SpeakerRequest):
    pass

class SpeakerUpdate(SpeakerRequest):
    prompt_text: Optional[str] = None
    prompt_audio_path: Optional[str] = None

class Speaker(SpeakerRequest):
    spk_success: bool
    create_at: datetime
    update_at: datetime
    
speakers: Dict[str, Speaker] = {}

@router.post("/spk", response_model=dict, status_code=200)
async def create_speaker(speaker: SpeakerCreate) -> dict:
    """Register a new speaker ID with provided audio and text prompt."""
    temp_file = None
    try:
        if speaker.zero_shot_spk_id in speakers:
            return {
                    "is_success": False,
                    "Error": f"Speaker ID already exists: {speaker.zero_shot_spk_id}",
                    "speaker info": None
                }
        # 音频文件处理
        if speaker.prompt_audio_path.startswith(('http://', 'https://')):
            try:
                logger.info(f"Downloading audio from URL: {speaker.prompt_audio_path}")
                temp_file = await download_audio_file(speaker.prompt_audio_path)
                prompt_speech_16k = load_wav(temp_file, 24000)
                logger.info("Successfully loaded prompt audio from URL")
            except Exception as e:
                logger.error(f"Error downloading audio: {str(e)}")
                raise AudioProcessingError(
                    status_code=400,
                    message="Error processing prompt audio file",
                    details={"error": str(e)}
                )
        else:
            if not os.path.exists(speaker.prompt_audio_path):
                logger.error(f"Audio file not found: {speaker.prompt_audio_path}")
                return {
                    "is_success": False,
                    "Error": f"Audio file not found: {speaker.prompt_audio_path}",
                    "speaker info": None
                }
            try:
                prompt_speech_16k = load_wav(speaker.prompt_audio_path, 16000)
            except Exception as e:
                logger.error(f"Error loading audio: {str(e)}")
                raise AudioProcessingError(
                    message="Error loading audio file",
                    status_code=400,
                    details={"error": str(e)}
                )
        
        try:
            spk_success = model.cosyvoice_model.add_zero_shot_spk(
                speaker.prompt_text,
                prompt_speech_16k,
                speaker.zero_shot_spk_id
            )
            if not spk_success:
                return {
                    "is_success": False,
                    "Error": "Failed to create speaker ID",
                    "speaker info": None
                }
            now = datetime.now()
            new_speaker = Speaker(
                **speaker.dict(),
                spk_success=spk_success,
                create_at=now,
                update_at=now
            )
            speakers[speaker.zero_shot_spk_id]=new_speaker
            return {
                    "is_success": True,
                    "Error": None,
                    "speaker info": new_speaker
                }
        except Exception as e:
            logger.error(f"Error registering speaker: {str(e)}")
            raise AudioProcessingError(
                    message="Failed to create speaker ID",
                    status_code=500,
                    details={"error": str(e)}
                )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")
                
@router.get("/spk", response_model=List[Speaker])
async def list_speakers() -> List[Speaker]:
    """List all speaker IDs"""
    return list(speakers.values())

@router.get("/spk/{zero_shot_spk_id}", response_model=dict)
async def get_speaker(zero_shot_spk_id: str = Path(..., title="Speaker ID")) -> dict:
    """Get speaker info"""
    if zero_shot_spk_id not in speakers:
        return {
                "is_success": False,
                "Error": f"Failed to get speaker info with {zero_shot_spk_id}",
                "speaker info": None
            }
    return {
            "is_success": True,
            "Error": None,
            "speaker info": speakers[zero_shot_spk_id]
        }