# Efficient 3D Recognition with Event-driven Spike Sparse Convolution

[Xuerui Qiu](https://scholar.google.com/citations?user=bMwW4e8AAAAJ&hl=zh-CN), [Man Yao](https://scholar.google.com/citations?user=eE4vvp0AAAAJ), [Jieyuan Zhang](https://scholar.google.com/citations?user=c8Qww6YAAAAJ&hl=zh-CN&oi=sra), [Yuhong Chou](https://scholar.google.com/citations?user=8CpWM4cAAAAJ&hl=zh-CN&oi=ao), [Ning Qiao](), [Shibo Zhou](), [Bo Xu](), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)

Institute of Automation, Chinese Academy of Sciences

This repo is the official implementation of [ Efficient 3D Recognition with Event-driven Spike Sparse Convolution](https://arxiv.org/abs/2412.07360). It currently concludes codes and models for the following tasks:

> **3D Classification**: See [Classification.md](/classify_seg/readme.md).\
> **3D Object Detection**: See [Detection.md](SDT_V3/Detection/Readme.md).\
> **3D Semantic Segmentation**: See [Segementation.md](/classify_seg/readme.md). \
> **Result_Visualizations**:  See [Visualizations.md](/Result_Visualizations/README.md).


:rocket:  :rocket:  :rocket: **News**:

- **Dec. 19, 2023**: Release the code for training and testing.

## Abstract
Spiking Neural Networks (SNNs) provide an energy-efficient way to extract 3D spatio-temporal features. Point clouds are sparse 3D spatial data, which suggests that SNNs should be well-suited for processing them. However, when applying SNNs to point clouds, they often exhibit limited performance and fewer application scenarios. We attribute this to inappropriate preprocessing and feature extraction methods. To address this issue, we first introduce the Spike Voxel Coding (SVC) scheme, which encodes the 3D point clouds into a sparse spike train space, reducing the storage requirements and saving time on point cloud preprocessing. Then, we propose a Spike Sparse Convolution (SSC) model for efficiently extracting 3D sparse point cloud features. Combining SVC and SSC, we design an efficient 3D SNN backbone (E-3DSNN), which is friendly with neuromorphic hardware. For instance, SSC can be implemented on neuromorphic chips with only minor modifications to the addressing function of vanilla spike convolution. Experiments on ModelNet40, KITTI, and Semantic KITTI datasets demonstrate that E-3DSNN achieves state-of-the-art (SOTA) results with remarkable efficiency. Notably, our E-3DSNN (1.87M) obtained 91.7\% top-1 accuracy on ModelNet40, surpassing the current best SNN baselines (14.3M) by 3.0\%. To our best knowledge, it is the first direct training 3D SNN backbone that can simultaneously handle various 3D computer vision tasks (e.g., classification, detection, and segmentation) with an event-driven nature.

## Results



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
