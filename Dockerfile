# HADRec 运行环境：只含代码和依赖，不含任何数据与模型权重
#
# 构建: docker build -t hadrec .
# 运行: docker run --gpus all --shm-size 16g -it \
#         -v $PWD/data:/workspace/HADRec/data \
#         -v $PWD/resources:/workspace/HADRec/resources \
#         -v $PWD/saved:/workspace/HADRec/saved \
#         -v $PWD/results:/workspace/HADRec/results \
#         hadrec
# 容器内: bash experiment/smoke_test.bash   # 先确认环境
#         bash experiment/llm_cls.bash      # 正式训练

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV PIP_NO_CACHE_DIR=1 \
    DS_BUILD_OPS=0 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl wget git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/HADRec

COPY requirements.txt .
RUN pip install setuptools wheel ninja \
    && grep -v '^deepspeed' requirements.txt | pip install -r /dev/stdin \
    && pip install --no-build-isolation $(grep '^deepspeed' requirements.txt)

COPY . .

CMD ["bash"]
