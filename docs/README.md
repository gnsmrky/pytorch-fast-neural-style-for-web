# Making the PyTorch fast-neural-style (FNS) to ONNX.js conversion work in browsers
This doc describes problems encountered to make the PyTorch [fast-neural-style (FNS) example](https://github.com/pytorch/examples/tree/master/fast_neural_style) work with ONNX.js in more technical backgrounds.  It provides a general guidance for any conversions between deep learning frameworks.

## The problems
With PyTorch v1.0 and [ONNX.js v0.1.3](https://github.com/Microsoft/onnxjs/tree/v0.1.3), there are few major problems:
1. Default ONNX opset level exported by PyTorch is `v9`, while ONNX.js is `v7`.
2. `InstanceNormalization` ONNX op is missing.  
_**Update:** Posted the "`InstanceNormalization` missing" issue [here](https://github.com/Microsoft/onnxjs/issues/18)_.  
_**Update:** `InstanceNormalization` is now supported in `master` branch by `cpu` and `wasm` backends (as of feb 15, 2019) with this [merged commit](https://github.com/Microsoft/onnxjs/pull/82#issuecomment-463867590).  The commit should be made available in next stable ONNX.js release._
3. `Upsample` ONNX op is missing

Usually a deep learning framework comes with various examples.  Running examples within the accompanied framework is usually ok.  Running examples **_in a different target framework_**, however, usually is **_not_**.  

## Making PyTorch trained model work in ONNX.js
One major technique is to minimize the changes in both PyTorch (source framework) and ONNX.js (target framework) as both frameworks are being updated frequently.  This is true particularly for ONNX.js as it is still in heavy development cycles.  

Thus, the following directions were followed:  
1. **Avoid changes to ONNX.js.**
   - As ONNX.js progresses, more ops will be supported.

2. **The only change for PyTorch is to change the default export opset level from `9` to `7`.**
   - In python environment, find the file `symbolic.py` for ONNX.  Search `_onnx_opset_version` within this file, change the number from `9` to `7`.  
   Change `_onnx_opset_version = 9` to `_onnx_opset_version = 7`
   - For example, in python 3.6 virtualenv for `pip` installed PyTorch (`pip install torch`), `symbolic.py` is usually located at:  
   `./lib/python3.6/site-packages/torch/onnx/symbolic.py`
   - Link to GitHub [PyTorch v1.0 onnx/symbolic.py#164](v1.0.0/torch/onnx/symbolic.py#L164)  
   
3. **Break down the un-supported `InstanceNormalization` and `Upsample` ops to basic ops _only for inference eval_**
   - The re-written model for inference eval is in `transformer_net_baseops.py`
   - Rewrite using the basic ops and make sure the ops run correctly in ONNX.js.  
      - PyTorch `torch.nn.InstanceNorm2d` layer, normally converted to `InstanceNormalization` op in ONNX, is being re-written by `InstanceNorm2d_ONNXJS()` class.  
      - PyTorch `torch.nn.functional.interpolate()` function, normally converted to `Upsample` op in ONNX, is being re-written by `_upsample_by_2()`.
   - Optimize and debug the re-written ops so the performance is optimal and runs in ONNX.js.  (Involves repeative tries with different supported ops and benchmark in ONNX.js, which also has its own bugs.)
4. **Make sure the pre-trained PyTorch weights and models (.pth files) can still be used.**
   - _So no re-training is needed!_
   - When training, the original `TransformerNet` class is still being used.
   - When running inference eval for exporting ONNX model files, `transformer_net_baseops` class is being used.

## Larger models with _much more_ nodes
The re-written model `transformer_net_baseops.py` replaces `InstanceNorm2d()` and `interpolate()` with basic ops.  The result is increased number of nodes in the graph.

- `InstanceNorm2d_ONNXJS()` adds many more ops.  As `InstanceNorm2d` op is called for most `Conv2d` outputs, the number of ops increases quite a bit.  Here is the comparison of the converted `torch.nn.InstanceNorm2d`, before and after re-written in basic ops:

   |Regular <b>InstanceNormalization</b> layer.   |<b>InstanceNormalization</b> layer composed of basic ops. |
   |:-:|:-:|
   | <img src="./imgs/instancenorm_baseops_01.png" height="500">  | <img src="./imgs/instancenorm_baseops_02.png" height="500">  |


- Similar goes for `interpolate()` being replaced by `_upsample_by_2()`.


   |Regular <b>Upsample</b> op   |<b>Upsample</b> op composed of basic ops |
   |:-:|:-:|
   | <img src="./imgs/upsample_baseops_01.png" height="300">  | <img src="./imgs/upsample_baseops_02.png" height="300">  |

## Before and after

This is what it looks like in entirety.  
<center>
<table align="center">
   <th> &nbsp; </th>
   <th> Regular </th>
   <th> Re-written </th>
   <tr>
      <td align="center">Number of ops</td>
      <td width="200" align="center">66</td>
      <td width="200" align="center">371</td>
   </tr>
   <tr>
      <td align="center">ONNX Graph <br/>(click to view the graph)</td>
      <td width="200" align="center"> <a href="./imgs/mosaic_onnxrt.onnx.png"><img src="./imgs/mosaic_onnxrt.onnx.png" height="600"> </a></td>
      <td width="200" align="center"> <a href="./imgs/mosaic_onnxjs.onnx.png"><img src="./imgs/mosaic_onnxjs.onnx.png" height="600"></a> </td>
   </tr>
</table>
</center>

## Even smaller model sizes
Training
```
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/candy.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/candy.jpg

python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg

```
