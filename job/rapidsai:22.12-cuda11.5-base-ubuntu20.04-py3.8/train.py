import os

from snowflake.snowpark import Session

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def get_token():
    with open('/snowflake/session/token', 'r') as f:
        return f.read()

def train():
    print(os.environ['SNOWFLAKE_MOUNTED_STAGE_PATH'])
    connection_params = {
        'host': os.environ['SNOWFLAKE_HOST'],
        'port': os.environ['SNOWFLAKE_PORT'],
        'protocol': 'https',
        'account': os.environ['SNOWFLAKE_ACCOUNT'],
        'authenticator': 'oauth',
        'token': get_token(),
        'role': 'SERVICESNOW_USER_ROLE',
        'warehouse': 'MADKINS',
        'database': 'SPCS_DEMO',
        'schema': 'PROD'
    }

    session = Session.builder.configs(connection_params).create()

    model_gpu = SentenceTransformer('all-mpnet-base-v2', device='cuda:0')

    # We'd use more examples in the real-world scenario.
    train_examples = [
        InputExample(
            texts=['My Martin 0-16NY is a beautiful piece of craftsmanship.', 'My guitar is a beautiful piece of craftsmanship.'], label=0.7
        ),
        InputExample(
            texts=['My Martin 0-16NY is a beautiful piece of craftsmanship.', 'James walked quickly through the forest.'], label=0.1
        ),
        InputExample(
            texts=['When I played with a Martin 0-16NY every chord sounded perfect', 'Sarah played her favorite songs on her guitar'], label=0.8
        ),
        InputExample(
            texts=['Fender has been making guitars for decades.', 'Guitar-making has a rich history spanning decades.'], label=0.8
        ),
        InputExample(
            texts=['Baking is a great way to relax.', 'Guitar is an important instrument for rock and roll bands'], label=0.1
        ),
        InputExample(
            texts=['My Martin 0-16NY is a beautiful piece of craftsmanship.', 'My guitar is a beautiful piece of craftsmanship.'], label=0.8
        ),
        InputExample(
            texts=['My Martin 0-16NY is a beautiful piece of craftsmanship.', 'James walked quickly through the forest.'], label=0.1
        ),
        InputExample(
            texts=['When I played with a Martin 0-16NY every chord sounded perfect', 'Sarah played her favorite songs on her guitar'], label=0.8
        ),
        InputExample(
            texts=['Fender has been making guitars for decades.', 'Guitar-making has a rich history spanning decades.'], label=0.8
        ),
        InputExample(
            texts=['Baking is a great way to relax.', 'Guitar is an important instrument for rock and roll bands'], label=0.1
        ),
        InputExample(
            texts=['My Martin 0-16NY is a beautiful piece of craftsmanship.', 'My guitar is a beautiful piece of craftsmanship.'], label=0.8
        ),
        InputExample(
            texts=['My Martin 0-16NY is a beautiful piece of craftsmanship.', 'James walked quickly through the forest.'], label=0.1
        ),
        InputExample(
            texts=['When I played with a Martin 0-16NY every chord sounded perfect', 'Sarah played her favorite songs on her guitar'], label=0.8
        ),
        InputExample(
            texts=['Fender has been making guitars for decades.', 'Guitar-making has a rich history spanning decades.'], label=0.8
        ),
        InputExample(
            texts=['Baking is a great way to relax.', 'Guitar is an important instrument for rock and roll bands'], label=0.1
        ),
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model_gpu)

    model_gpu.fit(train_objectives=[(train_dataloader, train_loss)], epochs=100, warmup_steps=100)

    model_gpu.save(os.environ['SNOWFLAKE_MOUNTED_STAGE_PATH']+"/fine-tuned-model")

    return None

if __name__ == '__main__':
    train()
