ARG BASE_IMAGE=nvcr.io/nvidia/rapidsai/rapidsai-core:cuda11.8-runtime-ubuntu22.04-py3.8
FROM $BASE_IMAGE

RUN mamba install -n rapids -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision torchaudio pytorch-cuda=11.8 bertopic \
    snowflake-snowpark-python -y