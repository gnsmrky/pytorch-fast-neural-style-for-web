
# Run PyTorch fast-neural-style example with ONNX.js in web browsers
A fork of PyTorch [fast-neural-style (FNS) example](https://github.com/pytorch/examples/tree/master/fast_neural_style).  The example has built-in onnx export that works with [ONNX Runtime](https://github.com/Microsoft/onnxruntime), but that's about it.  This fork is to modify the example so it runs with [ONNX.js](https://github.com/Microsoft/onnxjs) in web browsers.

Performance is not the key consideration here, but to make it runnable in target deep learning framework, such as web browsers with ONNX.js.  Many workarounds are needed.  This repository is to find out what it takes to make the model conversion a successful one.

It follows the following process:  
<p align="center"><b>PyTorch FNS example --> PyTorch model files (.pth) --> ONNX model files --> ONNX.js on web browsers</b></p>

As both PyTorch and ONNX.js are being updated frequently, to minimize the scope of change, _most changes happens in fast-neural-style example only_.

## Setup and convert pre-trained PyTorch model files (.pth)

1. Setup PyTorch - [PyTorch get started](https://pytorch.org/get-started/locally/)
   - This includes setting up CUDA if necessary.
2. Setup ONNX.
   - [ONNX](https://github.com/onnx/onnx) GitHub repository.
3. Clone this repository, download the pre-trained models.
   - `git clone https://github.com/gnsmrky/pytorch-fast-neural-style.git`
   - Run `download_saved_models.py` to download the pre-trained `.pth` models.  
   4 models will be downloaded and extracted to `saved_models` folder: `candy.pth`, `mosaic.pth`, `rain_princess.pth` and `udnie.pth`

4. Run inference eval and export the `.pth` model to `.onnx` files.  For example, to convert/export `mosaic.pth` to `mosaic.onnx`: 
   - nVidia GPU:  
   `python neural_style/neural_style.py eval --model saved_models/mosaic.pth --content-image images/content-images/amber.jpg --output-image amber_mosaic.jpg --export_onnx saved_onnx/mosaic.onnx --cuda 1`
   - CPU: specify `--cuda 0` in the above python command line.
   - The exported `.onnx` model file is saved in `saved_onnx` folder.

The generated `.onnx` file can then be inferenced by ONNX.js in supported web browsers.

## System and web browser resource considerations
When running inference eval on a resource limited systems, such as CPU + 8GB of RAM or GPU + 2GB VRAM, the eval may result in **`Segmentation fault (core dumped)`** error.  This is mainly due to insufficient memory when doing inference run.  PyTorch needs to run inference to build model graph.  One quick way around this is to reduce the content image size by specifying `--content-scale`.  Specify `--content-scale 2` would resize the content image to half for both width and height.  

In the above inference eval, `amber.jpg` is an image of size 1080x1080.  `--content-scale 2` would scale down the image size to 540x540.  
```
python neural_style/neural_style.py eval --model saved_models/mosaic.pth --content-image images/content-images/amber.jpg \
                                        --content-scale 2 --output-image amber_mosaic.jpg --export_onnx saved_onnx/mosaic.onnx --cuda 1
```

(Reduced content size does not create smaller `.onnx` model file.  It simply reduces the amount of resources needed for the needed inference run.  In the exported `.onnx` model files, only the sizes of input and output nodes are changed.)

## Eval-to-export to smaller sizes
When doing inference eval with ONNX.js, the available resource is even more limited in web browsers.  It is recommended to lower down the resolution even futher, to 128x128 and 256x256.

Content image `amber.jpg` has resolution of 1080x1080:
   - For target output size of 128x128, use `--content-scale 8.4375` (1080 / 128 = 8.4375)
   - For target output size of 256x256, use `--content-scale 4.21875`(1080 / 256 = 4.21875)


Eval and export `candy.pth` --> `candy_128x128.onnx` and `candy_256x256.onnx` in `saved_onnx` folder.
```
python neural_style/neural_style.py eval --model saved_models/candy.pth --content-image images/content-images/amber.jpg \
           --content-scale 8.4375 --output-image amber_candy_128.jpg --cuda 1 --export_onnx saved_onnx/candy_128x128.onnx
python neural_style/neural_style.py eval --model saved_models/candy.pth --content-image images/content-images/amber.jpg \
           --content-scale 4.21875 --output-image amber_candy_256.jpg --cuda 1 --export_onnx saved_onnx/candy_256x256.onnx
```

Same for the rest of pre-traine `.pth` model files.  
- `mosaic.pth` --> `mosaic_128x128.onnx` and - `mosaic_256x256.onnx`  
- `rain_princess.pth` --> `rain_princess_128x128.onnx` and `rain_princess_256x256.onnx`  
- `udnie.pth` --> `udnie_128x128.onnx` and `udnie_256x256.onnx`

## The problems and what it takes to make it work.
With PyTorch v1.0 and [ONNX.js v0.1.3](https://github.com/Microsoft/onnxjs/tree/v0.1.3), there are 2 major problems:
- Default ONNX opset level exported by PyTorch is `v9`, while ONNX.js is `v7`.
- 2 ops are missing in ONNX.js, `InstanceNorm` and `Upsample` ops.  
_Update: Posted the "`InstanceNorm` missing" issue [here](https://github.com/Microsoft/onnxjs/issues/18)_.  
_Update: `InstanceNorm` is now supported using `cpu` and `wasm` as of feb 15, 2019 with this [merged ommit](https://github.com/Microsoft/onnxjs/pull/82#issuecomment-463867590)_.
- Combined with few other incompatibilities between PyTorch and ONNX.js.

It is frustrating for a deep learning beginner to go through various frameworks, model formats, model conversions, and developing and deploying a deep learning application.  Usually a deep learning framework comes with various examples.  Running such examples within the accompanied framework is usually ok.  Running examples in another framework, however, requires model conversion and the knowledge about the target framework.

One major technique is to minimize the changes in both PyTorch (source framework) and ONNX.js (target framework) as both frameworks are being updated frequently.  This is true particularly for ONNX.js as it is still in heavy development cycles.  

Thus, the following technicques were used:  
1. The only change for PyTorch is to change the default export opset level from 9 to 7.
   - In python environment, find the file `symbolic.py` for ONNX.  Search `_onnx_opset_version` within this file, change the number from `9` to `7`.  
   Change `_onnx_opset_version = 9` to `_onnx_opset_version = 7`
   - For example, in python 3.6 virtualenv for `pip` installed PyTorch (`pip install torch`), `symbolic.py` is usually located at:  
   `./lib/python3.6/site-packages/torch/onnx/symbolic.py`
   - Link to GitHub [PyTorch v1.0 onnx/symbolic.py#164](v1.0.0/torch/onnx/symbolic.py#L164)  
   
2. Break down the un-supported `InstanceNorm` and `Upsample` ops to basic ops _only for inference eval_
   - The re-written model for inference eval is in `transformer_net_baseops.py`
   - Rewrite using the basic ops and make sure the ops run correctly in ONNX.js.  
      - `InstanceNorm2d_ONNXJS()` class replaces `InstanceNorm2d()` class  
      - `_upsample_by_2()` replaces `interpolate()` in `UpsampleConvLayer` class.
   - Optimize the re-written ops so the performance is optimal in ONNX.js.  (Involves repeative tries with different basic ops and benchmark in ONNX.js.)
3. Make sure the pre-trained PyTorch weights and models (.pth files) can still be used.
   * _So no re-training is needed!_
4. Avoid changes to ONNX.js.

## Even smaller model sizes
Training
```
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/candy.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/candy.jpg

python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg

```

----------
##### Below from original repo of PyTorch fast-nueral-style

# fast-neural-style :city_sunrise: :rocket:
This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0).

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## Requirements
The program is written in Python, and uses [pytorch](http://pytorch.org/), [scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop using saved models.

## Usage
Stylize image
```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Train model
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download).
* `--style-image`: path to style-image.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``neural_style/neural_style.py`` for other command line arguments. For training new models you might have to tune the values of `--content-weight` and `--style-weight`. The mosaic style model shown above was trained with `--content-weight 1e5` and `--style-weight 1e10`. The remaining 3 models were also trained with similar order of weight parameters with slight variation in the `--style-weight` (`5e10` or `1e11`).

## Models

Models for the examples shown below can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0) or by running the script ``download_saved_models.py``.

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">		
</div>

<div align='center'>
  <img src='images/style-images/mosaic.jpg' height="174px">
  <img src='images/output-images/amber-mosaic.jpg' height="174px">
  <img src='images/output-images/amber-candy.jpg' height="174px">
  <img src='images/style-images/candy.jpg' height="174px">
  <br>
  <img src='images/style-images/rain-princess-cropped.jpg' height="174px">
  <img src='images/output-images/amber-rain-princess.jpg' height="174px">
  <img src='images/output-images/amber-udnie.jpg' height="174px">
  <img src='images/style-images/udnie.jpg' height="174px">
</div>
