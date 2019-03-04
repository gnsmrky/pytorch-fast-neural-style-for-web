import torch
import onnx
from onnx import numpy_helper as nphelper

import argparse

parser = argparse.ArgumentParser(description='Count parameters in a PyTOrch model file.')
parser.add_argument('model_path', metavar='MODEL_PATH', type=str,
                    help='path to an PyTorch model file (.pth or .model)')

args = parser.parse_args()

total_params = 0

if args.model_path.lower().endswith((".pth", ".model")):
    state_dict = torch.load(args.model_path)
    for key in list(state_dict.keys()):
        w = state_dict[key]
        ss = w.shape

        num_param = 1
        for dim in ss:
            num_param *= dim
            
        total_params += num_param

elif args.model_path.lower().endswith(".onnx"):
    model = onnx.load(args.model_path)
    initializer = model.graph.initializer
    
    total_params = 0
    for param in initializer:
        w = nphelper.to_array(param)
        ss = w.shape

        num_param = 1
        for dim in ss:
            num_param *= dim
            
        total_params += num_param
        
print ("total parameters: {:,}".format (total_params))




