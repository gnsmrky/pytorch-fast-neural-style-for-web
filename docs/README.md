# Making the PyTorch fast-neural-style (FNS) to ONNX.js conversion work in browsers
This doc describes problems encountered to make the PyTorch [fast-neural-style (FNS) example](https://github.com/pytorch/examples/tree/master/fast_neural_style) work with ONNX.js in more technical backgrounds.  It provides a general guidance for any conversions between deep learning frameworks.

Usually a deep learning framework comes with various examples.  Running examples within the accompanied framework is usually ok.  Running examples **_in a different target framework_**, however, usually is **_not_**.  Knowing how to tweak the network to make it run in target framework greatly extends the deployment flexibility and scope coverage.

Goto [PyTorch fast-neural-style web benchmark](https://gnsmrky.github.io/pytorch-fast-neural-style-onnxjs/benchmark.html) as a quick demo with ONNX.js running on web browsers.

#### Quick links:
- [The problems](#the-problems)
- [Making PyTorch trained model work in ONNX.js](#making-pytorch-trained-model-work-in-onnxjs)
- [A quick summary of ops between frameworks](#a-quick-summary-of-ops-between-frameworks)
- [Break down unsupported ops and layers](#Break-down-unsupported-ops-and-layers)
   - [Re-write `torch.nn.InstanceNorm2d` op using base ops](#re-writetorchnninstancenorm2d-op-using-base-ops)
   - [Re-write `torch.nn.functional.interpolate()` op using base ops](#re-writetorchnnfunctionalinterpolate-op-using-base-ops)
   - [Re-write `torch.nn.ReflectionPad2d()` op using base ops](#re-write-torchnnreflectionpad2d-op-using-base-ops)
- [Before and after](#Before-and-after)
- [Train reduced models for even smaller model sizes](#Train-reduced-models-for-even-smaller-model-sizes)

## The problems
With PyTorch v1.0 and [ONNX.js v0.1.3](https://github.com/Microsoft/onnxjs/tree/v0.1.3), there are few op support/compatibility problems.  While PyTorch exports a sub-set of ONNX ops, ONNX.js supports even fewer ops than PyTorch.  This results in a much smaller ONNX op set in ONNX.js.
1. `InstanceNormalization` ONNX op support is missing.  
   _**Update:** Posted the "`InstanceNormalization` missing" issue [here](https://github.com/Microsoft/onnxjs/issues/18)_.  
_**Update:** `InstanceNormalization` is now supported in `master` branch by `cpu` and `wasm` backends (as of feb 15, 2019) with this [merged commit](https://github.com/Microsoft/onnxjs/pull/82#issuecomment-463867590).  The commit should be made available in next stable ONNX.js release._  
_**Update:** `InstanceNormalization` is supported in v0.1.4 `cpu` and `wasm` backend.  But 'wasm' backend has [issue #102](https://github.com/Microsoft/onnxjs/issues/102) about 'memory access out of bounds' error.  It's been_ fixed in [pull request #104](https://github.com/Microsoft/onnxjs/pull/104), so should be fixed in next release.
2. `Upsample` ONNX op support is missing.
3. `Pad` ONNX op support is missing in 'cpu' and 'wasm' backend.  To use the newly added `InstanceNormalization` in v0.1.4, had to manually implement `ZeroPad`.

## Making PyTorch trained model work in ONNX.js
One major technique is to minimize the changes in both PyTorch (source framework) and ONNX.js (target framework) as both frameworks are being updated frequently.  Particularly for ONNX.js as it is still in heavy development cycles.  _This is also true for any similar **conversion --> deployment** developments between 2 or more different frameworks._

Thus, the following directions were followed:  
1. **Avoid changes to ONNX.js.**
   - As ONNX.js progresses, more ops will be supported. However, do contribute issues to [ONNX.js issues in Github](https://github.com/Microsoft/onnxjs/issues).  The group is quite active and resposive to issues reported.
2. **Avoid changes to PyTorch.**
3. **Break down the un-supported `InstanceNormalization`, `Upsample` and `Pad` ops to basic ops _only for inference eval_.**
   - The re-written network model `TransformerNet_BaseOps` is in `transformer_net_baseops.py`, _**only for inference eval**_.  
   - Rewrite using the basic ops and make sure the ops run correctly in ONNX.js.  
      - PyTorch `torch.nn.InstanceNorm2d` layer class, normally converted to `InstanceNormalization` op in ONNX, is being re-written by `InstanceNorm2d_ONNXJS13()` class.  
      - PyTorch `torch.nn.functional.interpolate()` function, normally converted to `Upsample` op in ONNX, is being re-written by `_upsample_by_2()`.
   - Optimize and debug the re-written ops so the performance is optimal when running in ONNX.js.  (Involves repeatitive tries with different supported ops and benchmark in ONNX.js.)
3. **Make sure the pre-trained PyTorch weights and models (.pth files) can still be used.**
   - _So no re-training is needed!_
   - The original `TransformerNet` network is still being used for training.  All training process reamins the same as the original repo.

## A quick summary of ops between frameworks  
|PyTorch op (`torch.nn`) |Exported ONNX op<br/>(ONNXRuntime or WinML)|ONNX.js v0.1.3 op|ONNX.js v0.1.4 op|
|:--:|:--:|:--:|:--:|
|`InstanceNorm2d`|`InstanceNormalization`| not supported | &nbsp; Only `cpu`&nbsp;backend <br/> (`wasm`&nbsp;has&nbsp;[issue #102](https://github.com/Microsoft/onnxjs/issues/102)) |
| `functional.interpolate` | `Upsample` | not supported | not supported |
| `ReflectionPad2d` | `Pad` | Only&nbsp;`webgl`&nbsp;backend |  Only&nbsp;`webgl`&nbsp;backend|


## Break down unsupported ops and layers
The re-written model `TransformerNet_BaseOps` class in `transformer_net_baseops.py` replaces `InstanceNorm2d`, `interpolate` and `ReflectionPad2d` with basic ops.  The model weights are still being trained by the  `TransformerNet` class in the original PyTorch FNS code base.  Just the model being exported is being re-written using `TransformerNet_BaseOps`.

When exporting ONNX by doing inference run, add option `--target_framework` to target different ONNX.js versions.

Example to export ONNX model file for ONNX.js 0.1.3:  
<code>python neural_style/neural_style.py eval --content-scale 8.4375  --model saved_models/candy.pth --content-image images/content-images/amber.jpg --cuda 1 --output-image amber_candy_128x128.jpg --export_onnx saved_onnx/candy_128x128.onnx
<b>--target_framework ONNXJS_013</b>  </code>

Supported target frameworks:  
- `ONNXJS`: Default, latest ONNX.js version using `webgl` backend.  
- `ONNXJS_CPU`: ONNX.js v0.1.4 and above using `cpu` backend.  
- `ONNXJS_013`: ONNX.js v0.13 using `webgl` backend.  
- `ONNXRT`: Original PyTorch ONNX export.
- `PLAIDML`: Experimental PlaidML compatible ONNX model.  


### Re-write`torch.nn.InstanceNorm2d` op using base ops 
In `TransformerNet`, `InstanceNorm2d` op is called for most `Conv2d` outputs, the number of ops increases quite a bit.  Here is the comparison of the converted `torch.nn.InstanceNorm2d`, before and after re-written in basic ops:

|| Regular ONNX<br/><br/>| ONNX.js v0.1.3 <br/>(issue #53 workaround)|ONNX.js v0.1.4<br/>(issue #53 fixed)|ONNX.js v0.1.4<br/>(issue #53 fixed)|
|:-:|:-:|:-:|:-:|:-:|
| Runtime Backend | ONNXRuntime/WinML | `webgl` |`webgl`|`cpu`<br/>(`wasm` still has issue) |
| Layer Class |`IntanceNormalization` expoted by `torch.nn.InstanceNorm2d()`|`InstanceNorm2d_ONNXJS013()` in `TransformerNet_BaseOps` class|`InstanceNorm2d()` in `TransformerNet_BaseOps` class (No `Add` op) |`IntanceNormalization` expoted by `torch.nn.InstanceNorm2d()`|
| Base Ops used | n/a | `reshape` `mean` `sqrt` `mul` `div` `add`  | `reshape` `mean` `sqrt` `mul` `div` | n/a |
| Op graph| <a href="./imgs/instancenorm_baseops_01.png"><img src="./imgs/instancenorm_baseops_01.png" height="500"></a>  | <a href="./imgs/instancenorm_baseops_02.png"><img src="./imgs/instancenorm_baseops_02.png" height="500"></a>  | <a href="./imgs/instancenorm_baseops_03.png"><img src="./imgs/instancenorm_baseops_03.png" height="500"></a>  | <a href="./imgs/instancenorm_baseops_01.png"><img src="./imgs/instancenorm_baseops_01.png" height="500"></a> |

### Re-write`torch.nn.functional.interpolate()` op using base ops 
Similar goes for `interpolate()` being re-written by `_upsample_by_2()`.  

One note is that the `Upsample` used in PyTorch FNS is only up-scaling tensors by scale of 2 in both width (w) and height (h) using `nearest` method.  This can very easily be replaced be `reshape` and `concat` ops.


   || Regular ONNX<br/><br/>| ONNX.js v0.1.3 <br/>(issue #53 workaround)|ONNX.js v0.1.4<br/>(issue #53 fixed)|
   |:-:|:-:|:-:|:-:|
   | Runtime Backend | ONNXRuntime/WinML | `webgl` |`webgl` `cpu`|
   |Op function| `Upsample` exported by `torch.nn.functional.interpolate()`|`upsample_by_2_ONNXJS013()` in `TransformerNet_BaseOps` class|`upsample_by_2()` in `TransformerNet_BaseOps` class|
   | Base Ops used | n/a | `reshape` `concat` `add`  |  `reshape` `concat`|
   |Op graph| <a href="./imgs/upsample_baseops_01.png"><img src="./imgs/upsample_baseops_01.png" height="300"></a>  | <a href="./imgs/upsample_baseops_02.png"><img src="./imgs/upsample_baseops_02.png" height="300"></a>  | <a href="./imgs/upsample_baseops_03.png"><img src="./imgs/upsample_baseops_03.png" height="300"></a>  |

### Re-write `torch.nn.ReflectionPad2d` op using base ops
`Pad` op is not supported by ONNX.js `cpu` and `wasm` backend.  The `ZeroPad()` is being used to pad constant `0` instead.
   | | Regular ONNX| ONNX.js v0.1.3|ONNX.js v0.1.4|ONNX.js v0.1.4|
   |:-:|:-:|:-:|:-:|:-:|
   |Runtime Backend|ONNXRuntime/WinML|`webgl`|`webgl`| `cpu`|
   |Op funtion|`Pad` exported by `torch.nn.ReflectionPad2d()`|`Pad` exported by `torch.nn.ReflectionPad2d()`|`Pad` exported by `torch.nn.ReflectionPad2d()`|`ZeroPadding()` in `TransformerNet_BaseOps` class|
   |Base Ops used|n/a|n/a|n/a|`concat`|
   |Op graph|<a href="./imgs/pad_baseops_01.png"><img src="./imgs/pad_baseops_01.png" height='100'></a>|<a href="./imgs/pad_baseops_01.png"><img src="./imgs/pad_baseops_01.png" height='100'></a>|<a href="./imgs/pad_baseops_01.png" height='100'><img src="./imgs/pad_baseops_01.png" height='100'></a>|<a href="./imgs/pad_baseops_02.png"><img src="./imgs/pad_baseops_02.png" height='100'></a>|

## Before and after

This is what it looks like in entirety.  The more re-written ops results in increased number of ops in exported ONNX model file.
<center>
<table align="center">
   <th> &nbsp; </th>
   <th> Regular ONNX</th>
   <th> ONNX.js v0.1.3 </th>
   <th> ONNX.js v0.1.4</th>
   <th> ONNX.js v0.1.4</th>
   <tr>
      <td align="center">Runtime Backend</td>
      <td align="center">ONNXRuntime/WinML</td>
      <td align="center"><code>webgl</code></td>
      <td align="center"><code>webgl</code></td>
      <td align="center"><code>cpu</code></td>
   </tr>
   <tr>
      <td align="center">Number of ops</td>
      <td width="200" align="center">66</td>
      <td width="200" align="center">371</td>
      <td width="200" align="center">333</td>
      <td width="200" align="center">126</td>
   </tr>
   <tr>
      <td align="center">Ops summary</td>
      <td>Add: <b>5</b><br/>
         Constant: <b>2</b><br/>
         Conv: <b>16</b><br/>
         InstanceNorm:&nbsp;<b>15</b><br/>
         Pad: <b>16</b><br/>
         Relu: <b>10</b><br/>
         Upsample: <b>2</b></td>
      <td>Add: <b>54</b><br/>
         Concat: <b>4</b><br/>
         Constant: <b>100</b><br/>
         Conv: <b>16</b><br/>
         Div: <b>15</b><br/>
         Mul: <b>30</b><br/>
         Pad: <b>16</b><br/>
         ReduceMean:&nbsp;<b>30</b><br/>
         Relu: <b>10</b><br/>
         Reshape: <b>66</b><br/>
         Sqrt: <b>15</b><br/>
         Sub: <b>15</b></td>
      <td>Add: <b>35</b><br/>
Concat: <b>4</b><br/>
Constant: <b>81</b><br/>
Conv: <b>16</b><br/>
Div: <b>15</b><br/>
Mul: <b>30</b><br/>
Pad: <b>16</b><br/>
ReduceMean: <b>30</b><br/>
Relu: <b>10</b><br/>
Reshape: <b>66</b><br/>
Sqrt: <b>15</b><br/>
Sub: <b>15</b></td>
      <td>Add: <b>5</b><br/>
         Concat: <b>36</b><br/>
         Constant: <b>38</b><br/>
         Conv: <b>16</b><br/>
         InstanceNorm: <b>15</b><br/>
         Relu: <b>10</b><br/>
         Reshape: <b>6</b></td>
   </tr>
   <tr>
      <td align="center">ONNX Graph <br/>(click to view graph)</td>
      <td width="200" align="center"> <a href="./imgs/mosaic_onnxrt.onnx.png"><img src="./imgs/mosaic_onnxrt.onnx.png" height="600"> </a></td>
      <td width="200" align="center"> <a href="./imgs/mosaic_onnxjs013.onnx.png"><img src="./imgs/mosaic_onnxjs013.onnx.png" height="600"></a> </td>
      <td width="200" align="center"> <a href="./imgs/mosaic_onnxjs014_webgl.onnx.png"><img src="./imgs/mosaic_onnxjs014_webgl.onnx.png" height="600"></a> </td>
      <td width="200" align="center"> <a href="./imgs/mosaic_onnxjs014_cpu.onnx.png"><img src="./imgs/mosaic_onnxjs014_cpu.onnx.png" height="600"></a> </td>
   </tr>
</table>
</center>

## Train reduced models for even smaller model sizes
Default start number of channels is `32` as in the original PyTorch `TransformerNet` class.  It then grows to `64` and `128` channels in `Conv2D` layers.  The `ResNet` layer followed, consisted 2 `Conv2D` each, also has `128` channels each.  

`torch.nn.InstanceNorm2d` follows most `Conv2D` operations.  The re-written `torch.nn.InstanceNorm2d` has many element-wise operations, such as `add` and `mul` ops.  Ops like `cat` is also memory consuming.  This results in increasing memory usage considerably.  

By reducing the start number of channels to `16` reduces the model variables in the network to just about 1/4 of the original model size.  Set the start number of channels to `8` reduce it even further to 1/16.  In a resource constrained runtime environment like web browser, reduced model helps stability when running FNS inference.

The model can be re-trained with smaller start number of channels by specifying `--num-channels 16` when running training.  (Please refers to the original [PyTorch fast-neural-style exmaple](https://github.com/pytorch/examples/tree/master/fast_neural_style#usage) GitHub repo on how to setup the training.  This repo follows the same setup steps.)

The `neural_style.py` in this repo has the following changes:

1. Add support for `--num-channels` option.
2. The result model file is automatically created and componsed of the following:
   - Style picture name.
   - `num-channels` number.
   - `batch-size` number.
   - `epocs` number.
   - `content-weight` number.
   - `style-weight` number.
   - Example: for `Candy.jpg`, with `--num-channels 16`, `--batch-size 1`, `--epocs 2`, default `--content-weight` (`1e05`), default `--style weight` (`1e10`), the model file name is made automatically as:  
      `candy_nc16_b1_e2_1e05_1e10.model`

Here is a list of code snippets for training:
<table>
<th>Style file</th><th>Result model files<br/>auto saved in <code>saved_models_nc16</code> folder</th><th>Snippets for exporting ONNX model files <br/> (<code>saved_onnx_nc16</code> folder)</th>
<tr>
<td>candy.jpg</td>
<td>candy_nc16_b1_e2_1e05_1e10.model</td>
<td>python neural_style/neural_style.py train --dataset data/ <b>--batch-size 1</b> --epochs 2 --cuda 1  --style-image images/style-images/candy.jpg <b>--num-channels 16</b>  <b>--save-model-dir saved_models_nc16</b></td> 
</tr>
<tr>
<td>mosaic.jpg</td>
<td>mosaic_nc16_b1_e2_1e05_1e10.model</td>
<td>python neural_style/neural_style.py train --dataset data/ <b>--batch-size 1</b> --epochs 2 --cuda 1  --style-image images/style-images/mosaic.jpg <b>--num-channels 16</b>  <b>--save-model-dir saved_models_nc16</b></td> 
</tr>
<tr>
<td>rain-princess.jpg</td>
<td>rain-princess_nc16_b1_e2_1e05_1e10.model</td>
<td>python neural_style/neural_style.py train --dataset data/ <b>--batch-size 1</b> --epochs 2 --cuda 1  --style-image images/style-images/rain-princess.jpg <b>--num-channels 16</b>  <b>--save-model-dir saved_models_nc16</b></td> 
</tr>
<tr>
<td>udnie.jpg</td>
<td>udnie_nc16_b1_e2_1e05_1e10.model</td>
<td>python neural_style/neural_style.py train --dataset data/ <b>--batch-size 1</b> --epochs 2 --cuda 1  --style-image images/style-images/udnie.jpg <b>--num-channels 16</b>  <b>--save-model-dir saved_models_nc16</b></td> 
</tr>
</table>


Note:
1. When using smaller `--num-channels` at `16`, `8`, or `4`, the number of trainable variable reduced signicantly.  It is recommended to set `--batch-size` to `1`, instead of the default value at `4`.
2. Using nVidia GTX 1070, when `--batch-size 1` and `--epochs 2`, it takes approximately 4~5 hours for the training.
3. The trained model files with `--num-channels 16` are provided in `saved_models_nc16` folder.
4. The trained model files with `--num-channels 8` are provided in `saved_models_nc8` folder.
5. When training with `--num-channels` equal or smaller than `8`, it is recommended to set the `--style-size` size to `2` for a smaller style image to learn from.


