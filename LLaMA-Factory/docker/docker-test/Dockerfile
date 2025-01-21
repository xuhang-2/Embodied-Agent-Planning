FROM cr.infini-ai.com/infini-ai/ubuntu:22.04-20240429
WORKDIR /root
SHELL [ "/bin/bash", "-c" ]
ENV SHELL=/bin/bash

RUN apt update && \
    apt install git -y && \
    apt install vim -y && \
    /usr/local/miniconda3/condabin/conda create -n llamafactory python=3.10 -y && \
    source /usr/local/miniconda3/bin/activate llamafactory && \
    pip install \
        "transformers>=4.41.2,<=4.45.0" \
        "datasets>=2.16.0,<=2.21.0" \
        "accelerate>=0.30.1,<=0.33.0" \
        "peft>=0.11.1,<=0.12.0" \
        "trl>=0.8.6,<=0.9.6" \
        scipy \
        einops \
        sentencepiece \
        tiktoken \
        protobuf \
        uvicorn \
        pydantic \
        fastapi \
        sse-starlette \
        "matplotlib>=3.7.0" \
        fire \
        packaging \
        pyyaml \
        "numpy<2.0.0"