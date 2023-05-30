use database topic_modeling;
use schema prod;

--load model
put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/tritonserver:22.12-pyt-python-py3/model_repository/topic_modeling/1/model.py
@tm_stage/model_repository/topic_modeling/1/ auto_compress=false overwrite=true;

put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/tritonserver:22.12-pyt-python-py3/model_repository/topic_modeling/config.pbtxt
@tm_stage/model_repository/topic_modeling/ auto_compress=false overwrite=true;

--load train.py
put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/rapidsai:22.12-cuda11.5-base-ubuntu20.04-py3.8/train.py
@tm_stage auto_compress=false overwrite=true;

--load client.py
put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/tritonserver:22.12-py3-sdk/triton_client.py
@tm_stage auto_compress=false overwrite=true;

--load manifest files
put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/manifest.yaml @tm_stage auto_compress=false overwrite=true;

put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/rapidsai:22.12-cuda11.5-runtime-ubuntu20.04-py3.8/rapidsai_notebook_manifest.yaml
@tm_stage auto_compress=false overwrite=true;

put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/rapidsai:22.12-cuda11.5-base-ubuntu20.04-py3.8/transformers_train_manifest.yaml
@tm_stage auto_compress=false overwrite=true;

put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/tritonserver:22.12-pyt-python-py3/tritonserver_manifest.yaml
@tm_stage auto_compress=false overwrite=true;

put file:///Users/madkins/Documents/github_repos/scs-topic-modeling/tritonserver:22.12-py3-sdk/tritonclient_manifest.yaml
@tm_stage auto_compress=false overwrite=true;
