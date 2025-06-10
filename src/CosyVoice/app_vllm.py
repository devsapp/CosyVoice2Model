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

@app.post("/tts")
def text_to_speech(request: TTSRequest):
    try:
        if not os.path.exists(request.prompt_audio_path):
            raise HTTPException(status_code=400, detail="Prompt audio file not found")
        
        prompt_speech_16k = load_wav(request.prompt_audio_path, 16000)

        def generate(is_stream):
            audio_data = None
            chunk_num = 0

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

            for chunk in inference_generator:
                chunk_num += 1
                if is_stream:
                    # 立即返回每个音频块
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, chunk['tts_speech'], cosyvoice.sample_rate, format='wav')
                    yield buffer.getvalue()
                else:
                    # 累积音频数据
                    audio_data = torch.cat([audio_data, chunk['tts_speech']], dim=1) if audio_data is not None else chunk['tts_speech']

            if not is_stream and audio_data is not None:
                buffer = io.BytesIO()
                torchaudio.save(buffer, audio_data, cosyvoice.sample_rate, format='wav')
                yield buffer.getvalue()

            logger.info(f"Task completed, generated {chunk_num} chunks")

        if request.stream:
            return StreamingResponse(generate(True), media_type="audio/wav")
        else:
            audio_data = b''.join(generate(False))
            return Response(content=audio_data, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)