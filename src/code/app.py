import os
import torch
import torchaudio
import logging
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
LOAD_TRT = os.environ.get('LOAD_TRT', 'False').lower() == 'true'
FP16 = os.environ.get('FP16', 'False').lower() == 'true'
USE_FLOW_CACHE = os.environ.get('USE_FLOW_CACHE', 'False').lower() == 'true'

app = FastAPI()

class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    mode: str = Field(
        default='zero_shot',
        description="推理模式: zero_shot, cross_lingual, instruct2"
    )
    prompt_audio_path: str = Field(
        default='./asset/zero_shot_prompt.wav',
        description="自定义音频路径"
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

class TTS:
    def __init__(
        self,
        model_path: str,
        load_jit: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
        use_flow_cache: bool = True
    ):
        from cosyvoice.cli.cosyvoice import CosyVoice2
        
        logger.info(f"Initializing model with parameters:")
        logger.info(f"- Model path: {model_path}")
        logger.info(f"- Load JIT: {load_jit}")
        logger.info(f"- Load TRT: {load_trt}")
        logger.info(f"- FP16: {fp16}")
        logger.info(f"- Use flow cache: {use_flow_cache}")
        
        self.model = CosyVoice2(
            model_path,
            load_jit=load_jit,
            load_trt=load_trt,
            fp16=fp16,
            use_flow_cache=use_flow_cache
        )
        self.sample_rate = self.model.sample_rate
        logger.info("Model loaded successfully")

    def generate_stream(self, request: TTSRequest):
        """流式推理"""
        try:    
            # 定义分隔符    
            boundary = b"--frame"   

            # 发送多部分响应的头部  
            yield b"".join([    
                b"--", boundary, b"\r\n",   
                b"Content-Type: audio/wav\r\n\r\n"  
            ])  

            # 加载prompt音频    
            logger.info(f"Loading prompt audio from: {request.prompt_audio_path}")  
            prompt_speech = torchaudio.load(request.prompt_audio_path)[0]   

            # 获取生成器    
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

            # 流式返回音频数据  
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

                # 发送音频数据  
                yield wav_buffer.read() 

                # 发送分隔符    
                yield b"".join([    
                    b"\r\n--", boundary, b"\r\n",   
                    b"Content-Type: audio/wav\r\n\r\n"  
                ])  

            # 发送结束标记  
            yield b"".join([b"\r\n--", boundary, b"--\r\n"])    
            logger.info(f"Stream generation completed, total chunks: {chunk_count}")    

        except Exception as e:  
            logger.error(f"Stream generation error: {str(e)}")  
            raise   


    async def generate_non_stream(self, request: TTSRequest) -> bytes:
        """非流式推理"""
        try:
            # 加载prompt音频
            logger.info(f"Loading prompt audio from: {request.prompt_audio_path}")
            prompt_speech = torchaudio.load(request.prompt_audio_path)[0]

            # 生成音频
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

            # 转换为WAV格式
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
tts = TTS(
    model_path=MODEL_PATH,
    load_jit=LOAD_JIT,
    load_trt=LOAD_TRT,
    fp16=FP16,
    use_flow_cache=USE_FLOW_CACHE
)

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        logger.info(f"Received TTS request: {request.dict()}")
        
        if request.stream:
            # 流式响应
            logger.info("Starting streaming response")
            return StreamingResponse(
                tts.generate_stream(request),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Connection": "keep-alive",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 非流式响应
            logger.info("Starting non-streaming response")
            audio_data = await tts.generate_non_stream(request)
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
            "load_trt": LOAD_TRT,
            "fp16": FP16,
            "use_flow_cache": USE_FLOW_CACHE
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # 打印启动配置
    logger.info("Starting server with configuration:")
    logger.info(f"MODEL_PATH: {MODEL_PATH}")
    logger.info(f"LOAD_JIT: {LOAD_JIT}")
    logger.info(f"LOAD_TRT: {LOAD_TRT}")
    logger.info(f"FP16: {FP16}")
    logger.info(f"USE_FLOW_CACHE: {USE_FLOW_CACHE}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
