# Making the PyTorch to ONNX.js conversion work
This doc describes problems encountered to make the PyTorch [fast-neural-style (FNS) example](https://github.com/pytorch/examples/tree/master/fast_neural_style) work with ONNX.js in more technical backgrounds.  It provides a general guidance for any conversions between deep learning frameworks.

## The problems
With PyTorch v1.0 and [ONNX.js v0.1.3](https://github.com/Microsoft/onnxjs/tree/v0.1.3), there are 2 major problems:
- Default ONNX opset level exported by PyTorch is `v9`, while ONNX.js is `v7`.
- 2 ops are missing in ONNX.js, `InstanceNorm` and `Upsample` ops.  
_Update: Posted the "`InstanceNorm` missing" issue [here](https://github.com/Microsoft/onnxjs/issues/18)_.  
_Update: `InstanceNorm` is now supported in `master` branch by `cpu` and `wasm` backends (as of feb 15, 2019) with this [merged commit](https://github.com/Microsoft/onnxjs/pull/82#issuecomment-463867590).  The commit should be made available in next stable ONNX.js release._

It is frustrating for a deep learning beginner to go through various frameworks, model formats, model conversions, and developing and deploying a deep learning application.  Usually a deep learning framework comes with various examples.  Running examples within the accompanied framework is usually ok.  Running examples **_in a different target framework_**, however, usually is **_not_**.

## Making PyTorch trained model work in ONNX.js
One major technique is to minimize the changes in both PyTorch (source framework) and ONNX.js (target framework) as both frameworks are being updated frequently.  This is true particularly for ONNX.js as it is still in heavy development cycles.  

Thus, the following techniques were used:  
1. Avoid changes to ONNX.js.  
2. The only change for PyTorch is to change the default export opset level from `9` to `7`.
   - In python environment, find the file `symbolic.py` for ONNX.  Search `_onnx_opset_version` within this file, change the number from `9` to `7`.  
   Change `_onnx_opset_version = 9` to `_onnx_opset_version = 7`
   - For example, in python 3.6 virtualenv for `pip` installed PyTorch (`pip install torch`), `symbolic.py` is usually located at:  
   `./lib/python3.6/site-packages/torch/onnx/symbolic.py`
   - Link to GitHub [PyTorch v1.0 onnx/symbolic.py#164](v1.0.0/torch/onnx/symbolic.py#L164)  
   
3. Break down the un-supported `InstanceNorm` and `Upsample` ops to basic ops _only for inference eval_
   - The re-written model for inference eval is in `transformer_net_baseops.py`
   - Rewrite using the basic ops and make sure the ops run correctly in ONNX.js.  
      - `InstanceNorm2d_ONNXJS()` class replaces `InstanceNorm2d()` class  
      - `_upsample_by_2()` replaces `interpolate()` in `UpsampleConvLayer` class.
   - Optimize the re-written ops so the performance is optimal in ONNX.js.  (Involves repeative tries with different basic ops and benchmark in ONNX.js.)
4. Make sure the pre-trained PyTorch weights and models (.pth files) can still be used.
   * _So no re-training is needed!_

## Larger models
The re-written model replaces `InstanceNorm2d()` and `interpolate()` with basic ops.  The result is increased number of nodes in the graph.

`InstanceNorm2d_ONNXJS()` adds many more ops.  As `InstanceNorm2d` op is called for most `Conv2d` outputs, the number of ops increases quite a bit.  Here is the comparison of `InstanceNorm2d` before and after:

!!!PLACEHOLDER for InstanceNorm2d comparison!!!

Similar goes for `interpolate()` being replaced by `_upsample_by_2()`.

## Even smaller model sizes
Training
```
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/candy.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/candy.jpg

python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg

```
