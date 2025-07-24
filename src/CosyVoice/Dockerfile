FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing
RUN apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev && \
    apt-get clean && \
    git lfs install

# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    git pull && \
    git submodule update --init --recursive

RUN conda run -n ${VENV} conda install -y -c conda-forge pynini==2.1.5
COPY requirements.txt /workspace/
RUN conda run -n ${VENV} pip install -r /workspace/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN conda run -n ${VENV} pip install torch==2.7.0 torchaudio==2.7.0
RUN conda run -n ${VENV} pip install vllm==0.9.0

# 安装 funasr
RUN conda run -n ${VENV} pip install funasr

WORKDIR /workspace/CosyVoice

# COPY pretrained_models pretrained_models/
# COPY asset asset/

# 下载模型
RUN conda run -n ${VENV} python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
RUN conda run -n ${VENV} python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')"
RUN conda run -n ${VENV} python -c "from modelscope import snapshot_download; snapshot_download('iic/SenseVoiceSmall', local_dir='pretrained_models/SenseVoiceSmall')"

# 安装 ttsfrd
RUN cd pretrained_models/CosyVoice-ttsfrd/ && \
    conda run -n ${VENV} pip install ttsfrd_dependency-0.1-py3-none-any.whl && \
    conda run -n ${VENV} pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && \
    unzip resource.zip -d .

RUN conda run -n ${VENV} pip install numpy==1.24.3
RUN conda run -n ${VENV} pip install onnxruntime==1.15.1

# 添加 webui.py 脚本
COPY inference_project /workspace/CosyVoice/inference_project/

# ENV MODEL_DIR="pretrained_models/SenseVoiceSmall"

RUN touch /workspace/CosyVoice/cosyvoice/vllm/__init__.py
# 设置默认命令
CMD ["python3", "inference_project/app.py", "--port", "8000", "--model_dir", "pretrained_models/CosyVoice2-0.5B"]