import os
import torch
import torchaudio
import logging
import numpy as np
import random
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取配置
MODEL_PATH = os.environ.get('MODEL_PATH', 'pretrained_models/CosyVoice2-0.5B')
LOAD_JIT = os.environ.get('LOAD_JIT', 'False').lower() == 'true'
# LOAD_TRT = os.environ.get('LOAD_TRT', 'False').lower() == 'true'
FP16 = os.environ.get('FP16', 'False').lower() == 'true'
# USE_FLOW_CACHE = os.environ.get('USE_FLOW_CACHE', 'False').lower() == 'true'
DEFAULT_PROMPT_AUDIO = os.environ.get('DEFAULT_PROMPT_AUDIO', './asset/default_prompt.wav')

app = FastAPI()

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > 0.8:
        speech = speech / speech.abs().max() * 0.8
    speech = torch.concat([speech, torch.zeros(1, int(16000 * 0.2))], dim=1)
    return speech

def load_wav(file_path, sr):
    wav, _ = torchaudio.load(file_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if _ != sr:
        wav = torchaudio.functional.resample(wav, _, sr)
    return wav

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    mode: str = Field(
        default='zero_shot',
        description="推理模式: zero_shot, cross_lingual, instruct2"
    )
    prompt_audio_path: str = Field(
        default=DEFAULT_PROMPT_AUDIO,
        description="自定义prompt音频路径"
    )
    prompt_text: Optional[str] = Field(
        default='希望你以后能够做的比我还好呦。',
        description="prompt文本"
    )
    stream: bool = Field(
        default=False,
        description="是否使用流式推理"
    )
    speed: float = Field(
        default=1.0,
        description="语速"
    )
    seed: int = Field(
        default=0,
        description="随机推理种子"
    )

class TTS:
    def __init__(
        self,
        model_path: str,
        load_jit: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
        use_flow_cache: bool = True,
        default_prompt_audio: str = DEFAULT_PROMPT_AUDIO
    ):
        from cosyvoice.cli.cosyvoice import CosyVoice2
        
        logger.info(f"Initializing model with parameters:")
        logger.info(f"- Model path: {model_path}")
        logger.info(f"- Load JIT: {load_jit}")
        logger.info(f"- Load TRT: {load_trt}")
        logger.info(f"- FP16: {fp16}")
        logger.info(f"- Use flow cache: {use_flow_cache}")
        logger.info(f"- Default prompt audio: {default_prompt_audio}")
        
        self.model = CosyVoice2(
            model_path,
            load_jit=load_jit,
            load_trt=load_trt,
            fp16=fp16,
            use_flow_cache=use_flow_cache
        )
        self.sample_rate = self.model.sample_rate
        self.default_prompt_audio = default_prompt_audio
        logger.info("Model loaded successfully")

    def process_audio(self, audio_path):
        prompt_sr = 16000  # 设置目标采样率
        prompt_speech_16k = postprocess(load_wav(audio_path, prompt_sr))
        return prompt_speech_16k

    def generate_stream(self, request: TTSRequest):
        """流式推理"""
        try:    
            boundary = b"--frame"   

            yield b"".join([    
                b"--", boundary, b"\r\n",   
                b"Content-Type: audio/wav\r\n\r\n"  
            ])  

            set_all_random_seed(request.seed)
            prompt_speech = self.process_audio(request.prompt_audio_path)

            logger.info(f"Starting stream generation with mode: {request.mode}")    
            if request.mode == 'zero_shot': 
                generator = self.model.inference_zero_shot( 
                    request.text,   
                    request.prompt_text,    
                    prompt_speech,  
                    stream=True,    
                    speed=request.speed
                )   
            elif request.mode == 'cross_lingual':   
                generator = self.model.inference_cross_lingual( 
                    request.text,   
                    prompt_speech,  
                    stream=True,    
                    speed=request.speed
                )   
            elif request.mode == 'instruct2':   
                generator = self.model.inference_instruct2( 
                    request.text,   
                    request.prompt_text,    
                    prompt_speech,  
                    stream=True,    
                    speed=request.speed
                )   
            else:   
                raise ValueError(f"Unsupported mode: {request.mode}")   

            chunk_count = 0 
            for chunk in generator: 
                chunk_count += 1    
                wav_buffer = io.BytesIO()   
                torchaudio.save(    
                    wav_buffer, 
                    chunk['tts_speech'],    
                    self.sample_rate,   
                    format="wav"    
                )   
                wav_buffer.seek(0)  

                yield wav_buffer.read() 

                yield b"".join([    
                    b"\r\n--", boundary, b"\r\n",   
                    b"Content-Type: audio/wav\r\n\r\n"  
                ])  

            yield b"".join([b"\r\n--", boundary, b"--\r\n"])    
            logger.info(f"Stream generation completed, total chunks: {chunk_count}")    

        except Exception as e:  
            logger.error(f"Stream generation error: {str(e)}")  
            raise   

    async def generate_non_stream(self, request: TTSRequest) -> bytes:
        """非流式推理"""
        try:
            set_all_random_seed(request.seed)
            prompt_speech = self.process_audio(request.prompt_audio_path)

            logger.info(f"Starting non-stream generation with mode: {request.mode}")
            if request.mode == 'zero_shot':
                generator = self.model.inference_zero_shot(
                    request.text,
                    request.prompt_text,
                    prompt_speech,
                    stream=False,
                    speed=request.speed
                )
            elif request.mode == 'cross_lingual':
                generator = self.model.inference_cross_lingual(
                    request.text,
                    prompt_speech,
                    stream=False,
                    speed=request.speed
                )
            elif request.mode == 'instruct2':
                generator = self.model.inference_instruct2(
                    request.text,
                    request.prompt_text,
                    prompt_speech,
                    stream=False,
                    speed=request.speed
                )
            else:
                raise ValueError(f"Unsupported mode: {request.mode}")

            result = next(generator)
            logger.info("Audio generation completed")

            wav_buffer = io.BytesIO()
            torchaudio.save(
                wav_buffer,
                result['tts_speech'],
                self.sample_rate,
                format="wav"
            )
            return wav_buffer.getvalue()

        except Exception as e:
            logger.error(f"Non-stream generation error: {str(e)}")
            raise

# 初始化TTS模型
logger.info("Initializing TTS model...")
tts_stream = TTS(
    model_path=MODEL_PATH,
    load_jit=LOAD_JIT,
    load_trt=True,
    fp16=FP16,
    use_flow_cache=True,
    default_prompt_audio=DEFAULT_PROMPT_AUDIO
)
tts_non_stream = TTS(
    model_path=MODEL_PATH,
    load_jit=LOAD_JIT,
    load_trt=False,
    fp16=FP16,
    use_flow_cache=False,
    default_prompt_audio=DEFAULT_PROMPT_AUDIO
)
@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        logger.info(f"Received TTS request: {request.dict()}")
        
        if request.stream:
            logger.info("Starting streaming response")
            return StreamingResponse(
                tts_stream.generate_stream(request),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Connection": "keep-alive",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            logger.info("Starting non-streaming response")
            audio_data = await tts_non_stream.generate_non_stream(request)
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=output.wav"
                }
            )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "config": {
            "model_path": MODEL_PATH,
            "load_jit": LOAD_JIT,
            # "load_trt": LOAD_TRT,
            "fp16": FP16,
            # "use_flow_cache": USE_FLOW_CACHE,
            "default_prompt_audio": DEFAULT_PROMPT_AUDIO
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server with configuration:")
    logger.info(f"MODEL_PATH: {MODEL_PATH}")
    logger.info(f"LOAD_JIT: {LOAD_JIT}")
    # logger.info(f"LOAD_TRT: {LOAD_TRT}")
    logger.info(f"FP16: {FP16}")
    # logger.info(f"USE_FLOW_CACHE: {USE_FLOW_CACHE}")
    logger.info(f"DEFAULT_PROMPT_AUDIO: {DEFAULT_PROMPT_AUDIO}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)