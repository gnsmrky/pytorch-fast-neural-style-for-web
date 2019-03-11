# Making the PyTorch fast-neural-style (FNS) to ONNX.js conversion work in browsers
This doc describes problems encountered to make the PyTorch [fast-neural-style (FNS) example](https://github.com/pytorch/examples/tree/master/fast_neural_style) work with ONNX.js in more technical backgrounds.  It provides a general guidance for any conversions between deep learning frameworks.

Usually a deep learning framework comes with various examples.  Running examples within the accompanied framework is usually ok.  Running examples **_in a different target framework_**, however, usually is **_not_**.  Knowing how to tweak the network to make it run in target framework greatly extends the deployment flexibility and scope coverage.

Goto [PyTorch fast-neural-style web benchmark](https://gnsmrky.github.io/pytorch-fast-neural-style-onnxjs/benchmark.html) as a quick demo with ONNX.js running on web browsers.

## The problems
With PyTorch v1.0 and [ONNX.js v0.1.3](https://github.com/Microsoft/onnxjs/tree/v0.1.3), there are few op support/compatibility problems.  While PyTorch exports a sub-set of ONNX ops, ONNX.js supports even fewer ops than PyTorch.
1. `InstanceNormalization` ONNX op support is missing.  
   _**Update:** Posted the "`InstanceNormalization` missing" issue [here](https://github.com/Microsoft/onnxjs/issues/18)_.  
_**Update:** `InstanceNormalization` is now supported in `master` branch by `cpu` and `wasm` backends (as of feb 15, 2019) with this [merged commit](https://github.com/Microsoft/onnxjs/pull/82#issuecomment-463867590).  The commit should be made available in next stable ONNX.js release._  
_**Update:** `InstanceNormalization` is supported in v0.1.4.  But 'wasm' backend has [issue #102](https://github.com/Microsoft/onnxjs/issues/102) about 'memory access out of bounds' error.  It's been_ fixed in [pull request #104](https://github.com/Microsoft/onnxjs/pull/104), so should be fixed in next release.
2. `Pad` ONNX op support is missing in 'cpu' and 'wasm' backend.  To use the newly added `InstanceNormalization` in v0.1.4, had to manually implement `ZeroPad`.
3. `Upsample` ONNX op support is missing.

## Making PyTorch trained model work in ONNX.js
One major technique is to minimize the changes in both PyTorch (source framework) and ONNX.js (target framework) as both frameworks are being updated frequently.  Particularly for ONNX.js as it is still in heavy development cycles.  _This is also true for any similar developments between 2 or more different frameworks._

Thus, the following directions were followed:  
1. **Avoid changes to ONNX.js.**
   - As ONNX.js progresses, more ops will be supported. However, do contribute issues to [ONNX.js issues in Github](https://github.com/Microsoft/onnxjs/issues).  The group is quite active and resposive to issues reported.
2. **Avoid changes to PyTorch.**
3. **Break down the un-supported `InstanceNormalization`, `Upsample` and `P ops to basic ops _only for inference eval_**
   - The re-written model _**only for inference eval**_, `TransformerNet_BaseOps` is in `transformer_net_baseops.py`.  The training is still being done using the original `TransformerNet` network.
   - Rewrite using the basic ops and make sure the ops run correctly in ONNX.js.  
      - PyTorch `torch.nn.InstanceNorm2d` layer, normally converted to `InstanceNormalization` op in ONNX, is being re-written by `InstanceNorm2d_ONNXJS13()` class.  
      - PyTorch `torch.nn.functional.interpolate()` function, normally converted to `Upsample` op in ONNX, is being re-written by `_upsample_by_2()`.
   - Optimize and debug the re-written ops so the performance is optimal and runs in ONNX.js.  (Involves repeative tries with different supported ops and benchmark in ONNX.js, which also has its own bugs.)
3. **Make sure the pre-trained PyTorch weights and models (.pth files) can still be used.**
   - _So no re-training is needed!_
   - When training, the original `TransformerNet` class is still being used.
   - When running inference eval for exporting ONNX model files, `transformer_net_baseops` class is being used.

## A quick summary of op compatibility between frameworks:  
|PyTorch op (`torch.nn`) |ONNX op|ONNX.js v0.1.3 op|ONNX.js v0.1.4 op| Base Ops|
|:--:|:--:|:--:|:--:|:--:|:--:|
|`InstanceNorm2d`|`InstanceNormalization`| n/a | &nbsp;`cpu`&nbsp;backend  `wasm`&nbsp;has&nbsp;issue | `reshape` `mean` `sqrt` `mul` `add` `div`|
| `functional.interpolate` | `Upsample` | n/a | n/a | `reshape` `concat` `add` ONNX.js v0.1.3 has [issue #53](https://github.com/Microsoft/onnxjs/issues/53) |
| `ReflectionPad2d` | `Pad` | Only&nbsp;`webgl`&nbsp;backend | Only&nbsp;`webgl`&nbsp;backend | `cat` for ZeroPad |


## Same models with _much more_ supported nodes
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
