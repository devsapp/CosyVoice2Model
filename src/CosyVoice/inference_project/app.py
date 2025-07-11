import sys
import logging
from fastapi import FastAPI
import uvicorn
import model

# 设置日志
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 初始化 FastAPI
app = FastAPI()

# 初始化模型
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
from cosyvoice.cli.cosyvoice import CosyVoice2

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# 初始化 CosyVoice2 模型
model.cosyvoice_model = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=True,
    load_trt=True,
    load_vllm=True,
    fp16=True
)

# 注册路由
import cosyvoice_http, cosyvoice_ws, spk
app.include_router(cosyvoice_http.router, tags=["http"])
app.include_router(cosyvoice_ws.router, tags=["websocket"])
app.include_router(spk.router, tags=["speaker_id"])

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=60,
        limit_concurrency=100
    )