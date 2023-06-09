import json
import os
import numpy as np
import triton_python_backend_utils as pb_utils
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn, Tensor, device
from torch.utils.dlpack import to_dlpack, from_dlpack

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

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT1")
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

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

        def mean_pooling(model_output, attention_mask):
            model_output = torch.from_numpy(model_output[0])
            token_embeddings = model_output #First element of model_output contains all token embeddings
            attention_mask = torch.from_numpy(attention_mask)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask, input_mask_expanded, sum_mask

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        responses = []
        for request in requests:
            out_0 = pb_utils.get_input_tensor_by_name(request, "output_0")
            out_0 = from_dlpack(out_0.to_dlpack()).detach().cpu().numpy()
            out_1 = pb_utils.get_input_tensor_by_name(request, "output_1")
            out_1 = from_dlpack(out_1.to_dlpack()).detach().cpu().numpy()
            att = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
            # Perform pooling
            sentence_embeddings = mean_pooling([out_0, out_1], att)
            out_1 = F.normalize(sentence_embeddings[0], p=2, dim=1)
            # Normalize embeddings
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_1.numpy().astype(self.output1_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_1])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        end = datetime.now()
        # logger.log(f'triton full: {end-start}')
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
