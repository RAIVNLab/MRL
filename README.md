# [Matryoshka Representations for Adaptive Deployment](https://arxiv.org/abs/2205.13147)
Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi

This repository contains the code for training and evaluating Matryoshka Representation with ResNet-50 backbone, along with visualizations. Training code is built upon [FFCV](https://github.com/libffcv/ffcv-imagenet) modified for MRL. The repository is organized as follows:

1. Set up
2. Matryoshka Linear Layer
3. Training ResNet50 Models
4. Inference
5. Model Analysis
5. Retrieval Performance


## Set Up
Pip install the requirements file in this directory:
```
pip install -r requirements.txt
```
We use PyTorch Distributed Data Parallel shared over 2 A100 GPUs and FFCV for the dataloader. FFCV uses 8 A100 GPUs, therefore we linearly downscale the learning rate by 4.

### Preparing the dataset.
Download the standard Imagenet-1k train and val set, and store the validation set in root directory with folder named `val/`. Using same code snippet as [FFCV](https://github.com/libffcv/ffcv-imagenet), generate the dataset from standard imagenet train set with the following command (`IMAGENET_DIR` should point to a PyTorch style [ImageNet dataset](https://github.com/MadryLab/pytorch-imagenet-dataset)):

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

#### Robustness Datasets

We evaluate our trained models on 4 robustness datasets- ImageNet-V2/A/R/Sketch. For these datasets, we don't need FFCV, therefore, we just use PyTorch. Please refer to the respective repositories for additional documentation and download the datasets in the root directory. 

1. [ImageNetV2_pytorch](https://github.com/modestyachts/ImageNetV2_pytorch)
2. [ImageNetA](https://github.com/hendrycks/natural-adv-examples)
3. [ImageNetR](https://github.com/hendrycks/imagenet-r)
4. [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

## Matryoshka Linear Layer
`MRL.py` provides MRL linear layer, and it can be instantiated very easily using the following
```
nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024, 2048]
fc_layer = MRL_Linear_Layer(nesting_list, num_classes=1000, efficient=efficient)
```
Where, `nesting_list` is the list of dimensions we want to nest on, `num_classes` is the number of output feature and efficient flag if we want to do MRL-E.

## [Training ResNet50 models](train/)
`cd train/`. While training, we dump  model ckpts, training logs and run command inside a folder. `$WRITE_DIR` is same variable that we used while creating the dataset. 

### Training Fixed Feature Baseline

```bash 
export CUDA_VISIBLE_DEVICES=0,1

python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.fixed_feature=2048 --data.train_dataset=$WRITE_DIR/train_500_0.50_90.ffcv --data.val_dataset=$WRITE_DIR/val_500_uncompressed.ffcv --data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 --dist.world_size=2 --training.distributed=1 --lr.lr=0.425
```

### Training MRL model

```bash 
export CUDA_VISIBLE_DEVICES=0,1

python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.mrl=1 --data.train_dataset=$WRITE_DIR/train_500_0.50_90.ffcv --data.val_dataset=$WRITE_DIR/val_500_uncompressed.ffcv --data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 --dist.world_size=2 --training.distributed=1 --lr.lr=0.425
```

By default, we start nesting from 8 dimensions (That is 2^3). In case we want to nest representations from 16 dimensions, add an additional flag of 
```
--model.nesting_start=4
```

### Training MRL-E model

```bash 
export CUDA_VISIBLE_DEVICES=0,1

python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.efficient=1 --data.train_dataset=$WRITE_DIR/train_500_0.50_90.ffcv --data.val_dataset=$WRITE_DIR/val_500_uncompressed.ffcv --data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 --dist.world_size=2 --training.distributed=1 --lr.lr=0.425
```

By default, we start nesting from 8 dimensions (That is 2^3). In case we want to nest representations from 16 dimensions, add an additional flag of 
```
--model.nesting_start=4
```

## [Inference on Trained Models](inference/)

### Classification performance
`cd inference/` . To evaluate the model, run the following command; argument in brackets are optional. To evaluate the Fixed Feature Baseline, just pass `--rep_size <dim>` to evaluate particular size dim. Script is also capable of using pytorch evaluation of standard imagenet-1k validation set (V1). To evaluate our uploaded checkpoints, please pass `--old_ckpt`. Our model checkpoints can be found [here](https://drive.google.com/drive/folders/1IEfJk4xp-sPEKvKn6eKAUzvoRV8ho2vq?usp=sharing). 

```python
python pytorch_eval.py --path <final_weigth.pt> --dataset <V2/A/Sketch/R/V1> [--tta] [--mrl] [--efficient] [--rep_size <dim>] [--old_ckpt]
```

In paper we only consider rep sizes in  `[8, 16, 32, 64, 128, 256, 512, 1024, 2048]`, therefore to evaluate on other rep. sizes change the variable `NESTING_LIST` in file `pytorch_eval.py`. 


## [Model Analysis](model_analysis/)
`cd model_analysis/` Here we provide 4 jupyter notebooks which contain performance visualization such as GradCAM images (for checkpoint models), superclass performance, model cascades and oracle upper bound. Please refer to its documentation [here](model_analysis/README.md).  

## Retrieval Performance


## Citation
If you find this project useful in your research, please consider citing:
```
@article{Kusupati2022MatryoshkaRF,
  title={Matryoshka Representations for Adaptive Deployment},
  author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham M. Kakade and Prateek Jain and Ali Farhadi},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.13147}
}
```
