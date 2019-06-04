Recursive Visual Attention in Visual Dialog
====================================

This repository contains the code for the following paper:

* Yulei Niu, Hanwang Zhang, Manli Zhang, Jianhong Zhang, Zhiwu Lu, Ji-Rong Wen, *Recursive Visual Attention in Visual Dialog*. In CVPR, 2019. ([PDF](https://arxiv.org/pdf/1812.02664.pdf))

```
@InProceedings{niu2019recursive,
    author = {Niu, Yulei and Zhang, Hanwang and Zhang, Manli and Zhang, Jianhong and Lu, Zhiwu and Wen, Ji-Rong},
    title = {Recursive Visual Attention in Visual Dialog},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}

```

This code is reimplemented as a fork of [batra-mlp-lab/visdial-challenge-starter-pytorch][6].


Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.0, and provides out of the box support with CUDA 9 and CuDNN 7. Anaconda/Miniconda is the recommended to set up this codebase: 

### Anaconda or Miniconda

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][1].
2. Clone this repository and create an environment:

```shell
git clone https://www.github.com/yuleiniu/rva
conda create -n visdial-ch python=3.6

# activate the environment and install all dependencies
conda activate visdial-ch
cd rva/
pip install -r requirements.txt

# install this codebase as a package in development version
python setup.py develop
```


Download Data
-------------

1. Download the VisDial v1.0 dialog json files from [here][3] and keep it under `$PROJECT_ROOT/data` directory, for default arguments to work effectively.

2. Get the word counts for VisDial v1.0 train split [here][4]. They are used to build the vocabulary.

3. [batra-mlp-lab][6] provides pre-extracted image features of VisDial v1.0 images, using a Faster-RCNN pre-trained on Visual Genome. If you wish to extract your own image features, skip this step and download VisDial v1.0 images from [here][3] instead. Extracted features for v1.0 train, val and test are available for download at these links. Note that these files do not contain the bounding box information.

  * [`features_faster_rcnn_x101_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5): Bottom-up features of 36 proposals from images of `train` split.
  * [`features_faster_rcnn_x101_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5): Bottom-up features of 36 proposals from images of `val` split.
  * [`features_faster_rcnn_x101_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5): Bottom-up features of 36 proposals from images of `test` split.

4. [batra-mlp-lab][6] also provides pre-extracted FC7 features from VGG16.

  * [`features_vgg16_fc7_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_train.h5): VGG16 FC7 features from images of `train` split.
  * [`features_vgg16_fc7_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_val.h5): VGG16 FC7 features from images of `val` split.
  * [`features_vgg16_fc7_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_test.h5): VGG16 FC7 features from images of `test` split.

5. Download the GloVe pretrained word vectors from [here][12], and keep `glove.6B.300d.txt` under `$PROJECT_ROOT/data` directory.

Extracting Features (Optional)
-------------

### With Docker (Optional)
For Dockerfile, please refer to [batra-mlp-lab/visdial-challenge-starter-pytorch][8].

### Without Docker (Optional)

0. Set up opencv, [cocoapi][9] and [Detectron][10].

1. Prepare the [MSCOCO][11] and [Flickr][3] images.

2. Extract visual features.
```shell
python ./data/extract_features_detectron.py --image-root /path/to/MSCOCO/train2014/ /path/to/MSCOCO/val2014/ --save-path /path/to/feature --split train # Bottom-up features of 36 proposals from images of train split.
python ./data/extract_features_detectron.py --image-root /path/to/Flickr/VisualDialog_val2018 --save-path /path/to/feature --split val # Bottom-up features of 36 proposals from images of val split.
python ./data/extract_features_detectron.py --image-root /path/to/Flickr/VisualDialog_test2018 --save-path /path/to/feature --split test # Bottom-up features of 36 proposals from images of test split.
```

Initializing GloVe Word Embeddings
--------------
Simply run 
```shell
python data/init_glove.py
```


Training
--------

Train the model provided in this repository as:

```shell
python train.py --config-yml configs/rva.yml --gpu-ids 0 # provide more ids for multi-GPU execution other args...
```

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. Refer [visdialch/utils/checkpointing.py][7] for more details on how checkpointing is managed.

### Logging

We use [Tensorboard][2] for logging training progress. Recommended: execute `tensorboard --logdir /path/to/save_dir --port 8008` and visit `localhost:8008` in the browser.


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```shell
python evaluate.py --config-yml /path/to/config.yml --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0
```

This will generate an EvalAI submission file, and report metrics from the [Visual Dialog paper][5] (Mean reciprocal rank, R@{1, 5, 10}, Mean rank), and Normalized Discounted Cumulative Gain (NDCG), introduced in the first Visual Dialog Challenge (in 2018).

The metrics reported here would be the same as those reported through EvalAI by making a submission in `val` phase. To generate a submission file for `test-std` or `test-challenge` phase, replace `--split val` with `--split test`.


[1]: https://conda.io/docs/user-guide/install/download.html
[2]: https://www.github.com/lanpa/tensorboardX
[3]: https://visualdialog.org/data
[4]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[5]: https://arxiv.org/abs/1611.08669
[6]: https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
[7]: https://www.github.com/yuleiniu/rva/blob/master/visdialch/utils/checkpointing.py
[8]: https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch#docker
[9]: https://www.github.com/cocodataset/cocoapi
[10]: https://www.github.com/facebookresearch/Detectron
[11]: http://cocodataset.org/#download
[12]: http://nlp.stanford.edu/data/glove.6B.zip