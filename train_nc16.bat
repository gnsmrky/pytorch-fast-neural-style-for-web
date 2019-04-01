rem python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models_nc16 --style-image images/style-images/candy_256x256.jpg
rem python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models_nc16 --style-image images/style-images/candy_256x256.jpg

rem python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models_nc16 --style-image images/style-images/mosaic.jpg
rem python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models_nc16 --style-image images/style-images/mosaic.jpg

rem python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models_nc16 --style-image images/style-images/rain-princess.jpg
rem python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models_nc16 --style-image images/style-images/rain-princess.jpg

rem python neural_style/neural_style.py train --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e09 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg
python neural_style/neural_style.py train --batch-size 1 --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 5e09 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg
python neural_style/neural_style.py train --batch-size 1 --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg
python neural_style/neural_style.py train --batch-size 1 --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 5e10 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg
python neural_style/neural_style.py train --batch-size 1 --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e11 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg

python neural_style/neural_style.py train --batch-size 2 --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 5e09 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg
python neural_style/neural_style.py train --batch-size 2 --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg
python neural_style/neural_style.py train --batch-size 2 --num-channels 16 --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e11 --save-model-dir saved_models_nc16 --style-image images/style-images/udnie.jpg