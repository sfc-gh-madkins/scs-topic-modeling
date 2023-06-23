FROM nvcr.io/nvidia/rapidsai/rapidsai:22.12-cuda11.5-base-ubuntu20.04-py3.8

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -U --no-cache-dir sentence_transformers snowflake-snowpark-python[pandas]

WORKDIR /
