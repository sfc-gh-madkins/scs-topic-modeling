FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -U --no-cache-dir sentence_transformers

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models"]
