# [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
_Aditya Kusupati*, Gantavya Bhatt*, Aniket Rege*, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi_

Learned representations are used in multiple downstream tasks like web-scale search & classification. However, they are flat & rigid -- Information is diffused across dimensions and cannot be adaptively deployed without large post-hoc overhead. We fix both of these issues with **Matryoshka Representation Learning** (MRL)🪆. 

<p align="center">
<img src="./images/mrl-teaser.jpeg" width="512"/>
</p>

This repository contains code to train, evaluate, and analyze Matryoshka Representations with a ResNet50 backbone. The training pipeline utilizes efficient [FFCV](https://github.com/libffcv/ffcv-imagenet) dataloaders modified for MRL. The repository is organized as follows:

1. Set up
2. Matryoshka Linear Layer
3. Training ResNet50 Models
4. Inference
5. Model Analysis
5. Retrieval


## Set Up
Pip install the requirements file in this directory. Note that a python3 distribution is required:
```
pip3 install -r requirements.txt
```

### Preparing the Dataset
Following the ImageNet training pipeline of [FFCV](https://github.com/libffcv/ffcv-imagenet) for ResNet50, generate the dataset with the following command (`IMAGENET_DIR` should point to a PyTorch style [ImageNet dataset](https://github.com/MadryLab/pytorch-imagenet-dataset)):

```bash
# Required environmental variables for the script:
cd train/
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded, 90% raw pixel values
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
```
Note that we prepare the dataset with the following FFCV configuration:
* ResNet-50 training: 50% JPEG 500px side length (*train_500_0.50_90.ffcv*)
* ResNet-50 evaluation: 0% JPEG 500px side length (*val_500_uncompressed.ffcv*)
## Matryoshka Linear Layer
We make only a minor modification to the ResNet50 architecture via the MRL linear layer, defined in `MRL.py`, which can be instantiated as:
```
nesting_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
fc_layer = MRL_Linear_Layer(nesting_list, num_classes=1000, efficient=efficient)
```
Where `nesting_list` is the list of representation sizes we wish to train on, `num_classes` is the number of output features, and the `efficient` flag is to train MRL-E.

## [Training ResNet50 models](train/)
<p align="center">
<img src="./images/mrl-r50-accuracy.jpeg" width="784"/>
</p>

We use PyTorch Distributed Data Parallel shared over 2 A100 GPUs and FFCV dataloaders. FFCV utilizes 8 A100 GPUs, therefore we linearly downscale the learning rate by 4 to compensate. We utilize the `rn50_40_epochs.yaml` configuration file provided by FFCV to train MRL ResNet50 models for 40 epochs.
While training, we dump  model ckpts and training logs by default. `$WRITE_DIR` is same variable used to create the dataset. 

### Training Fixed Feature Baseline

```bash 
export CUDA_VISIBLE_DEVICES=0,1

python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.fixed_feature=2048 \
--data.train_dataset=$WRITE_DIR/train_500_0.50_90.ffcv --data.val_dataset=$WRITE_DIR/val_500_uncompressed.ffcv \
--data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
--dist.world_size=2 --training.distributed=1 --lr.lr=0.425
```

### Training MRL model

```bash 
export CUDA_VISIBLE_DEVICES=0,1

python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.mrl=1 \
--data.train_dataset=$WRITE_DIR/train_500_0.50_90.ffcv --data.val_dataset=$WRITE_DIR/val_500_uncompressed.ffcv \
--data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
--dist.world_size=2 --training.distributed=1 --lr.lr=0.425
```

### Training MRL-E model

```bash 
export CUDA_VISIBLE_DEVICES=0,1

python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.efficient=1 \
--data.train_dataset=$WRITE_DIR/train_500_0.50_90.ffcv --data.val_dataset=$WRITE_DIR/val_500_uncompressed.ffcv \
--data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
--dist.world_size=2 --training.distributed=1 --lr.lr=0.425
```

By default, we start nesting from rep. size = 8 (i.e. $2^3$). We provide flexibility in starting nesting, for example from rep. size = 16, with the `nesting_start` flag as: 
```
# to start nesting from d=16
--model.nesting_start=4
```

## [Inference on Trained Models](inference/)

### Classification performance
To evaluate our models, we utilize the `pytorch_inference.py` script; arguments in brackets are optional. This script is also able to evaluate the standard Imagenet-1K validation set (V1). To evaluate the Fixed Feature (FF) Baseline, pass `--rep_size <dim>` flag to evaluate a particular representation size. For example, to evaluate an FF model with rep. size = 512:

```python
cd inference

python pytorch_inference.py --path <final_weight.pt> --dataset <V2/A/Sketch/R/V1> --rep_size 512
```

Similarly, to evaluate MRL models, pass the `--mrl` flag (add `--efficient` for MRL-E). Note that for MRL models, the `rep_size` flag is not required. The general form of the command to evaluate trained models is:

```python
cd inference

python pytorch_inference.py --path <final_weight.pt> --dataset <V2/A/Sketch/R/V1> \
[--tta] [--mrl] [--efficient] [--rep_size <dim>] [--old_ckpt] [--save_logits] \
[--save_softmax] [--save_gt] [--save_predictions]
```

The `save_*` flags are useful for downstream [model analysis](model_analysis). Our script is able to perform "test time augmentation (tta)" during evaluation with the `--tta` flag. Note that the classification results reported in the paper are without tta, and tta is only used for adaptive classification using model cascades. Please refer to [model analysis](model_analysis) for further details.


Lastly, to evaluate our uploaded checkpoints (ResNet50), please additionally use the `--old_ckpt` flag. Our model checkpoints can be found [here](https://drive.google.com/drive/folders/1IEfJk4xp-sPEKvKn6eKAUzvoRV8ho2vq?usp=sharing), and are arranged according to the training routine. The model naming convention is such that `r50_mrl1_e0_ff2048.pt` corresponds to the model trained with MRL (here "e" refers to efficient) and `r50_mrl0_e0_ff256.pt` corresponds to the model with rep. size = 256 and trained without MRL. In the paper we only consider $rep. size \in  [8, 16, 32, 64, 128, 256, 512, 1024, 2048]$. To evaluate on other rep. sizes, change the variable `NESTING_LIST` in `pytorch_eval.py`. For a detailed description, please run `python pytorch_inference.py --help`.

#### Robustness Datasets

We also evaluate our trained models on four robustness datasets: ImageNetV2/A/R/Sketch. Note that for evaluation, we utilized PyTorch dataloaders. Please refer to their respective repositories for additional documentation and download the datasets in the root directory. 

1. [ImageNetV2_pytorch](https://github.com/modestyachts/ImageNetV2_pytorch)
2. [ImageNetA](https://github.com/hendrycks/natural-adv-examples)
3. [ImageNetR](https://github.com/hendrycks/imagenet-r)
4. [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)


## [Model Analysis](model_analysis/)
`cd model_analysis` 

We provide four Jupyter notebooks which contain performance visualization via GradCAM images (for checkpoint models), superclass performance, model cascades and oracle upper bound. Please refer to detailed documentation [here](model_analysis/).  

## [Retrieval](retrieval/)
We carry out image retrieval on ImageNet-1K with two query sets, ImageNet-1K validation set and ImageNetV2. We also created [ImageNet-4K](imagenet-4k) to evaluate MRL image retrieval in an out-of-distribution setting, with its validation set used as query set. A detailed description of the retrieval pipeline is provided [here](retrieval/). 

In an attempt to achieve optimal compute-accuracy tradeoff, we carry out **Adaptive Retrieval** by retrieving a $k=$ 200 length neighbors shortlist with lower dimension $D_s$ and reranking with higher dimension $D_r$. We also provide a simple cascading policy to automate the choice of appropriate $D_s$ and $D_r$, which we call **Funnel Retrieval**. We retrieve a shortlist at $D_s$ and then re-rank the shortlist five times while simultaneously increasing $D_r$ (rerank cascade) and decreasing the shortlist length $k$ (shortlist cascade), which resembles a funnel structure. With both of these techniques, we are able to match the Top-1 accuracy (%) of retrieval with $D_s=$ 2048 with 128$\times$ less MFLOPs/Query on ImageNet-1K.

## Citation
If you find this project useful in your research, please consider citing:
```
@inproceedings{kusupati2022matryoshka,
  title     = {Matryoshka Representation Learning},
  author    = {Kusupati, Aditya and Bhatt, Gantavya and Rege, Aniket and Wallingford, Matthew and Sinha, Aditya and Ramanujan, Vivek and Howard-Snyder, William and Chen, Kaifeng and Kakade, Sham and Jain, Prateek and others},
  title     = {Matryoshka Representation Learning.},
  booktitle = {Advances in Neural Information Processing Systems},
  month     = {December},
  year      = {2022},
}
```
