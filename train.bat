python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e9 --style-image images/style-images/candy.jpg --save-model-dir saved_models/candy_nf16
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --style-image images/style-images/candy.jpg --save-model-dir saved_models/candy_nf16

python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e9  --style-image images/style-images/mosaic.jpg --save-model-dir saved_models/mosaic_nf16
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --style-image images/style-images/mosaic.jpg --save-model-dir saved_models/mosaic_nf16

python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e9  --style-image images/style-images/rain-princess.jpg --save-model-dir saved_models/rain-princess_nf16
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --style-image images/style-images/rain-princess.jpg --save-model-dir saved_models/rain-princess_nf16

python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e9  --style-image images/style-images/udnie.jpg --save-model-dir saved_models/udnie_nf16
python neural_style/neural_style.py train --dataset data/ --epochs 2 --cuda 1 --content-weight 1e5 --style-weight 1e10 --style-image images/style-images/udnie.jpg --save-model-dir saved_models/udnie_nf16