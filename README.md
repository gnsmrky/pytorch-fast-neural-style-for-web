
# PyTorch fast-neural-style example to run with ONNX.js in web browsers
A fork of PyTorch fast-neural-style example.  PyTorch has built-in onnx export that works with ONNX Runtime, but that's pretty much it.  This fork is to modify the example so it runs with [ONNX.js](https://github.com/Microsoft/onnxjs) in web browsrs.

Performance is not the key consideration here, but to make it runnable in target deep learning framework, such as web browsers with ONNX.js.  There are many workarounds needed.  This repository is to find out what it takes to make the model conversion a successful one.

It follows the following process:
PyTorch (source framework) --> PyTorch model (.pth or .model) --> ONNX models --> ONNX.js (target framework)

As both PyTorch and ONNX.js are being updated frequently, to minimize the scope of change, most changes happens in fast-neural-style example only.  The network model is updated to make it runnable in ONNX.js.

## Setup and convert pre-train model files

1. Setup PyTorch - [PyTorch get started](https://pytorch.org/get-started/locally/)
   - This includes setting up CUDA
2. Setup ONNX.
   - [ONNX](https://github.com/onnx/onnx) repository.
3. Clone this repository, download the pre-trained models.
   - `git clone https://github.com/gnsmrky/pytorch-fast-neural-style.git`
   - Run `download_saved_models.py` to download the pre-trained `.pth` models.  4 models will be downloaded and extracted to `saved_models` folder, `candy.pth`, `mosaic.pth`, `rain_princess.pth` and `udnie.pth`

4. Run inference eval and export the `.pth` model to `.onnx` files.  For example, to convert `mosaic.pth` to `mosaic.onnx`: 
   - NV GPU: `python neural_style/neural_style.py eval --model saved_models/mosaic.pth --content-image images/content-images/amber.jpg --output-image amber_mosaic.jpg --export_onnx saved_onnx/mosaic.onnx --cuda 1`
   - CPU: specify `--cuda 0` in the above python command line.
   - The exported `.onnx` model file is saved in `saved_onnx` folder.

The generated `.onnx` file can then be inferenced by ONNX.js in supported web browsers.

## System and web browser resource considerations
When running inference eval on a resource limited systems, such as CPU + 8GB of RAM, the eval may result in seg fault.  This is mainly due to insufficient memory.  One quick way around this is to reduce the content image size by specifying `--content-scale`.  Specify `--content-scale 2` would resize the content image to half for both width and height.  

In the above inference eval, `amber.jpg` is an image of 1080x1080.  `--content-scale 2` would resize down the image to 540x540.
```
python neural_style/neural_style.py eval --model saved_models/mosaic.pth --content-image images/content-images/amber.jpg --content-scale 2 --output-image amber_mosaic.jpg --export_onnx saved_onnx/mosaic.onnx --cuda 1
```

(Reduced content size does not result in reduced `.onnx` model file sizes.  It simply reduces the amount of resources needed for the needed inferencing eval run.  In the exported `.onnx` model files, only the sizes for input and output nodes are changed.)

## Eval-to-export in different sizes
If using `amber.jpg`, which has resolution 1080x1080:
   - For target output size of 128x128, use `--content-scale 8.4375` (1080 / 128 = 8.4375)
   - For target output size of 256x256, use `--content-scale 4.21875`(1080 / 256 = 4.21875)


Eval and export `candy.pth` --> `candy_128x128.onnx` and `candy_256x256.onnx`
```
python neural_style/neural_style.py eval --model saved_models/candy.pth --content-image images/content-images/amber.jpg --content-scale 8.4375 --output-image amber_candy_128.jpg --cuda 1 --export_onnx saved_onnx/candy_128x128.onnx
python neural_style/neural_style.py eval --model saved_models/candy.pth --content-image images/content-images/amber.jpg --content-scale 4.21875 --output-image amber_candy_256.jpg --cuda 1 --export_onnx saved_onnx/candy_256x256.onnx
```

Eval and export `mosaic.pth` --> `mosaic_128x128.onnx` and `mosaic_256x256.onnx`
```
python neural_style/neural_style.py eval --model saved_models/mosaic.pth --content-image images/content-images/amber.jpg --content-scale 8.4375 --output-image amber_mosaic_128.jpg --cuda 1 --export_onnx saved_onnx/mosaic_128x128.onnx
python neural_style/neural_style.py eval --model saved_models/mosaic.pth --content-image images/content-images/amber.jpg --content-scale 4.21875 --output-image ambermosaic_256.jpg --cuda 1 --export_onnx saved_onnx/mosaic_256x256.onnx
```

Eval and export `rain_princess.pth` --> `rain_princess_128x128.onnx` and `rain_princess_256x256.onnx`
```
python neural_style/neural_style.py eval --model saved_models/rain_princess.pth --content-image images/content-images/amber.jpg --content-scale 8.4375 --output-image amber_rain_princess_128.jpg --cuda 1 --export_onnx saved_onnx/rain_princess_128x128.onnx
python neural_style/neural_style.py eval --model saved_models/rain_princess.pth --content-image images/content-images/amber.jpg --content-scale 4.21875 --output-image amber_rain_princess_256.jpg --cuda 1 --export_onnx saved_onnx/rain_princess_256x256.onnx
```

Eval and export `udnie.pth` --> `udnie_128x128.onnx` and `udnie_256x256.onnx`
```
python neural_style/neural_style.py eval --model saved_models/udnie.pth --content-image images/content-images/amber.jpg --content-scale 8.4375 --output-image amber_udnie_128.jpg --cuda 1 --export_onnx saved_onnx/rain_princess_128x128.onnx
python neural_style/neural_style.py eval --model saved_models/udnie.pth --content-image images/content-images/amber.jpg --content-scale 4.21875 --output-image amber_udnie_256.jpg --cuda 1 --export_onnx saved_onnx/udnie_256x256.onnx
```

Training
```
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/candy.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/candy.jpg

python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models --style-image images/style-images/mosaic.jpg

```

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
