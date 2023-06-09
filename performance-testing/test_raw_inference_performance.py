import argparse
from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
from datetime import datetime

from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from snowflake.snowpark import Session
import numpy as np
import os
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def get_token():
    with open('/snowflake/session/token', 'r') as f:
        return f.read()

def run(device):

    model_name = "topic_modeling"

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
        REVIEW
    from
        "TOPIC_MODELING"."PROD"."MUSIC_STORE_REVIEWS"
    limit 100000

    '''

    df = session.sql(query).to_pandas()
    df[0] = df.index
    df = df[df.columns[::-1]]

    model_gpu = SentenceTransformer('all-mpnet-base-v2', device=f'cuda:{device}')
    start = datetime.now()
    emb = model_gpu.encode(df['REVIEW'].tolist(), batch_size=1000)
    end = datetime.now()
    print(end-start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentence Embeddings Program')
    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    run(args.device)