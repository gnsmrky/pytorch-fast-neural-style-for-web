import onnx
from onnx import numpy_helper as nphelper
import argparse

parser = argparse.ArgumentParser(description='Count parameters in an ONNX model file.')
parser.add_argument('model_path', metavar='MODEL_PATH', type=str,
                    help='path to an ONNX model file (.onnx)')

args = parser.parse_args()

#MODEL_PATH = "saved_onnx/candy_128x128.onnx"
#MODEL_PATH = "saved_onnx_nc16/candy_nc16_128x128.onnx"

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




