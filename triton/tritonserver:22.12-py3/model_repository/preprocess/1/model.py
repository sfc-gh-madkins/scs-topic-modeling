import json
import os
import numpy as np
import triton_python_backend_utils as pb_utils
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import nn, Tensor, device

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        logger = pb_utils.Logger

        model_path = os.environ['SNOWFLAKE_MOUNTED_STAGE_PATH']+"/notebook/topic-modeling/fine-tuned-model"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.decode = lambda x: x.decode('utf-8', 'ignore')

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        self.max_batch_size = self.model_config['max_batch_size']
        self.model_instance_device_id = json.loads(args['model_instance_device_id'])

        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        start = datetime.now()
        logger = pb_utils.Logger

        cuda_device = f'cuda:{self.model_instance_device_id}'

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        responses = []
        for request in requests:
            out_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy().astype(np.int64)

            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()
            in_1 = np.vectorize(self.decode)(in_1.flatten()).tolist()
            model_input = in_1
            start2 = datetime.now()

            # Tokenize sentences
            tokens = self.tokenizer(model_input, return_tensors="np")
            tokens = {k: v for k, v in tokens.items()}

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype(self.output0_dtype))

            outputs = [out_tensor_0]
            for input_name in self.tokenizer.model_input_names:
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=outputs)
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        end = datetime.now()
        # logger.log(f'preprocess: {end-start}')
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
