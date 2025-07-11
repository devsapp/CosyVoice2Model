# 用于解决循环依赖的问题
import sys
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
from cosyvoice.cli.cosyvoice import CosyVoice2

cosyvoice_model = None