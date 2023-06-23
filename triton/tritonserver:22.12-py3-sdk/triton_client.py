import logging

import pandas as pd
import numpy as np

from fastapi import FastAPI
import uvicorn

import tritonclient.http as httpclient
from tritonclient.utils import *

from datetime import datetime


app = FastAPI()


@app.get("/")
async def get_root():
    return {"Hello": "World"}

@app.post("/identity")
async def post_identity(payload: dict):
    return payload

@app.post("/inference_snowpark_triton")
async def post_inference_topic_modeling(payload: dict):
    start = datetime.now()
    logger = logging.getLogger("examples.identity_python.client")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    #parse input payload
    #logger.info(f"input: {payload}")
    df = pd.DataFrame(payload['data'])

    int_obj = np.array(df[0].apply(lambda x: [x]).tolist(), dtype=np.int64)
    text_obj = np.array(df[1].apply(lambda x: [x]).tolist(), dtype="object")

    input = [
        httpclient.InferInput("INPUT0", int_obj.shape, np_to_triton_dtype(int_obj.dtype)),
        httpclient.InferInput("INPUT1", text_obj.shape, np_to_triton_dtype(text_obj.dtype)),
    ]
    input[0].set_data_from_numpy(int_obj)
    input[1].set_data_from_numpy(text_obj)

    output = [
        httpclient.InferRequestedOutput("OUTPUT0"),
        httpclient.InferRequestedOutput("OUTPUT1"),
    ]
    start2 = datetime.now()
    with httpclient.InferenceServerClient(url="localhost:8000") as tritonclient:
        result = tritonclient.infer(model_name='topic_modeling', inputs=input, outputs=output)
    end2 = datetime.now()
    logger.info(f"triton full: {end2-start2}")

    df = pd.DataFrame(columns=[['OUTPUT0', 'OUTPUT1']])
    df['OUTPUT0'] = result.as_numpy(name='OUTPUT0').flatten()
    df['OUTPUT0'] = df['OUTPUT0'].apply(lambda x: x.astype(np.int64))

    return_list = []
    for i,arr in enumerate(result.as_numpy(name='OUTPUT1')):
        return_list.append([i, arr.tolist()])

    end = datetime.now()
    logger.info(f"client full: {end-start}")
    return {'data': return_list}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
