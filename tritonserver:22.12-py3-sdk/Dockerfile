FROM nvcr.io/nvidia/tritonserver:22.12-py3-sdk

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -U --no-cache-dir pandas fastapi uvicorn tritonclient[all]

ENTRYPOINT ["uvicorn", "triton_client:app", "--host", "0.0.0.0", "--port", "8080"]
