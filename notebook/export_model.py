from transformers.convert_graph_to_onnx import convert
import onnx_graphsurgeon as gs
import onnx

convert(framework="pt", model="/workspace/all-mpnet-base-v2", output="/workspace/all-mpnet-base-v2.onnx", opset=16)

graph = gs.import_onnx(onnx.load("/workspace/all-mpnet-base-v2.onnx"))
for inp in graph.inputs:
    inp.shape[0] = gs.Tensor.DYNAMIC
for out in graph.outputs:
    out.shape[0] = gs.Tensor.DYNAMIC
onnx.save(gs.export_onnx(graph.fold_constants().cleanup()),"/workspace/dynamic.onnx")









# # torch.save(model, "torch2.pth")

# # Tokenize sentences
# # encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# # # Compute token embeddings
# with torch.no_grad():
#     # model_output = model(**encoded_input)
#     torch.onnx.export(model,               # model being run
#                     encoded_input,                         # model input (or a tuple for multiple inputs)
#                     "torch_export.onnx",   # where to save the model (can be a file or file-like object)
#                     export_params=True,        # store the trained parameter weights inside the model file
#                     opset_version=16,          # the ONNX version to export the model to
#                     do_constant_folding=True,  # whether to execute constant folding for optimization
#                     #   input_names = ['INPUT0', 'INPUT1'],   # the model's input names
#                     #   output_names = ['OUTPUT0', 'OUTPUT1'], # the model's output names
#                     #   dynamic_axes={'INPUT0' : {0 : 'batch_size'},    # variable length axes
#                     #                 'INPUT1' : {0 : 'batch_size'},    # variable length axes
#                     #                 'OUTPUT0' : {0 : 'batch_size'},    # variable length axes
#                     #                 'OUTPUT1' : {0 : 'batch_size'}}
#                                     )