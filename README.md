# Efficient 3D Recognition with Event-driven Spike Sparse Convolution

[Xuerui Qiu](https://scholar.google.com/citations?user=bMwW4e8AAAAJ&hl=zh-CN), [Man Yao*](https://scholar.google.com/citations?user=eE4vvp0AAAAJ),[Jieyuan Zhang](https://scholar.google.com/citations?user=c8Qww6YAAAAJ&hl=zh-CN&oi=sra), [Yuhong Chou](https://scholar.google.com/citations?user=8CpWM4cAAAAJ&hl=zh-CN&oi=ao), [Ning Qiao](), [Shibo Zhou](), [Bo Xu](), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)

Institute of Automation, Chinese Academy of Sciences

This repo is the official implementation of [ Efficient 3D Recognition with Event-driven Spike Sparse Convolution](https://arxiv.org/abs/2412.07360). It currently concludes codes and models for the following tasks:

> **3D Classification**: See [Classification.md](SDT_V3/Classification/Model_Base/Train_Base.md).\
> **3D Object Detection**: See [Detection.md](SDT_V3/Detection/Readme.md).\
> **3D Semantic Segmentation**: See [Segementation.md](SDT_V3/Segmentation/Readme.md). \


:rocket:  :rocket:  :rocket: **News**:

- **Dec. 19, 2023**: Release the code for training and testing.

## Abstract
Spiking Neural Networks (SNNs) provide an energy-efficient way to extract 3D spatio-temporal features. Point clouds are sparse 3D spatial data, which suggests that SNNs should be well-suited for processing them. However, when applying SNNs to point clouds, they often exhibit limited performance and fewer application scenarios. We attribute this to inappropriate preprocessing and feature extraction methods. To address this issue, we first introduce the Spike Voxel Coding (SVC) scheme, which encodes the 3D point clouds into a sparse spike train space, reducing the storage requirements and saving time on point cloud preprocessing. Then, we propose a Spike Sparse Convolution (SSC) model for efficiently extracting 3D sparse point cloud features. Combining SVC and SSC, we design an efficient 3D SNN backbone (E-3DSNN), which is friendly with neuromorphic hardware. For instance, SSC can be implemented on neuromorphic chips with only minor modifications to the addressing function of vanilla spike convolution. Experiments on ModelNet40, KITTI, and Semantic KITTI datasets demonstrate that E-3DSNN achieves state-of-the-art (SOTA) results with remarkable efficiency. Notably, our E-3DSNN (1.87M) obtained 91.7\% top-1 accuracy on ModelNet40, surpassing the current best SNN baselines (14.3M) by 3.0\%. To our best knowledge, it is the first direct training 3D SNN backbone that can simultaneously handle various 3D computer vision tasks (e.g., classification, detection, and segmentation) with an event-driven nature.

## Results


## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

### Conda Environment

```bash
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```

## Data Preparation


### SemanticKITTI
- Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
- Link dataset to codebase.
```bash
# SEMANTIC_KITTI_DIR: the directory of SemanticKITTI dataset.
# |- SEMANTIC_KITTI_DIR
#   |- dataset
#     |- sequences
#       |- 00
#       |- 01
#       |- ...

mkdir -p data
ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
```

### nuScenes
- Download the official [NuScene](https://www.nuscenes.org/nuscenes#download) dataset (with Lidar Segmentation) and organize the downloaded files as follows:
```bash
NUSCENES_DIR
│── samples
│── sweeps
│── lidarseg
...
│── v1.0-trainval 
│── v1.0-test
```
- Run information preprocessing code (modified from OpenPCDet) for nuScenes as follows:
```bash
# NUSCENES_DIR: the directory of downloaded nuScenes dataset.
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
# MAX_SWEEPS: Max number of sweeps. Default: 10.
pip install nuscenes-devkit pyquaternion
python pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py --dataset_root ${NUSCENES_DIR} --output_root ${PROCESSED_NUSCENES_DIR} --max_sweeps ${MAX_SWEEPS} --with_camera
```
- (Alternative) Our preprocess nuScenes information data can also be downloaded [[here](
https://huggingface.co/datasets/Pointcept/nuscenes-compressed)] (only processed information, still need to download raw dataset and link to the folder), please agree the official license before download it.

- Link raw dataset to processed NuScene dataset folder:
```bash
# NUSCENES_DIR: the directory of downloaded nuScenes dataset.
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
ln -s ${NUSCENES_DIR} {PROCESSED_NUSCENES_DIR}/raw
```
then the processed nuscenes folder is organized as follows:
```bash
nuscene
|── raw
    │── samples
    │── sweeps
    │── lidarseg
    ...
    │── v1.0-trainval
    │── v1.0-test
|── info
```

- Link processed dataset to codebase.
```bash
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
mkdir data
ln -s ${PROCESSED_NUSCENES_DIR} ${CODEBASE_DIR}/data/nuscenes
```


### ModelNet
- Download [modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and unzip
- Link dataset to the codebase.
```bash
mkdir -p data
ln -s ${MODELNET_DIR} ${CODEBASE_DIR}/data/modelnet40_normal_resampled
```

## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard, and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base
```
**Resume training from checkpoint.** If the training process is interrupted by accident, the following script can resume training from a given checkpoint.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```

### Testing
During training, model evaluation is performed on point clouds after grid sampling (voxelization), providing an initial assessment of model performance. However, to obtain precise evaluation results, testing is **essential**. The testing process involves subsampling a dense point cloud into a sequence of voxelized point clouds, ensuring comprehensive coverage of all points. These sub-results are then predicted and collected to form a complete prediction of the entire point cloud. This approach yields  higher evaluation results compared to simply mapping/interpolating the prediction. In addition, our testing code supports TTA (test time augmentation) testing, which further enhances the stability of evaluation performance.

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n semseg-pt-v2m2-0-base -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base weight=exp/scannet/semseg-pt-v2m2-0-base/model/model_best.pth
```

The TTA can be disabled by replace `data.test.test_cfg.aug_transform = [...]` with:

```python
data = dict(
    train = dict(...),
    val = dict(...),
    test = dict(
        ...,
        test_cfg = dict(
            ...,
            aug_transform = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ]
        )
    )
)
```

## Contact Information

```
@article{qiu2024efficient,
  title={Efficient 3D Recognition with Event-driven Spike Sparse Convolution},
  author={Qiu, Xuerui and Yao, Man and Zhang, Jieyuan and Chou, Yuhong and Qiao, Ning and Zhou, Shibo and Xu, Bo and Li, Guoqi},
  journal={arXiv preprint arXiv:2412.07360},
  year={2024}
}
```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `qiuxuerui2024@ia.ac.cn`.

## Acknowledgement
Our project are based on [Pointcept](https://github.com/Pointcept/Pointcept) and [Openpcdet](https://github.com/open-mmlab/OpenPCDet). Thanks for their wonderful work.
