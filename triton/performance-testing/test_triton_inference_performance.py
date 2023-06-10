from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from snowflake.snowpark import Session
import numpy as np
import os
from datetime import datetime

model_name = "topic_modeling"

def get_token():
    with open('/snowflake/session/token', 'r') as f:
        return f.read()

connection_params = {
    'host': os.environ['SNOWFLAKE_HOST'],
    'port': os.environ['SNOWFLAKE_PORT'],
    'protocol': 'https',
    'account': os.environ['SNOWFLAKE_ACCOUNT'],
    'authenticator': 'oauth',
    'token': get_token(),
    'role': 'SERVICESNOW_USER_ROLE',
    'warehouse': 'MADKINS',
    'database': 'TOPIC_MODELING',
    'schema': 'PROD'
}

session = Session.builder.configs(connection_params).create()

query = '''
select
    REVIEWER
from
    "TOPIC_MODELING"."PROD"."MUSIC_STORE_REVIEWS_TEST"

'''

df = session.sql(query).to_pandas()
df[0] = df.index
df = df[df.columns[::-1]]

payload = {'data': df.values.tolist()}
df.head()

payload = {'data': df.values.tolist()}

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
start = datetime.now()
with httpclient.InferenceServerClient(url="localhost:8000") as tritonclient:
    result = tritonclient.infer(model_name='topic_modeling', inputs=input, outputs=output)
end = datetime.now()
print(end-start)