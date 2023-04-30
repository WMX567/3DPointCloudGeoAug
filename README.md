## 3D Point Cloud Completion with Geometric-Aware Adversarial Augmentation
This repository contains the official implementation for the paper "Point Cloud Completion with Geometric-Aware Adversarial Augmentation", ICPR 2022.

### Models

Point Completion Network [paper](https://arxiv.org/abs/1808.00671)

### Dependencies
* Python 3.7.9
* PyTorch 1.7.0
* CUDA 10.1.243
* Open3d
* pyemd
* PyTorch3D

### Dataset
ShapeNet [link](https://shapenet.org/)

### Run Experiments
Please change the paths, information of dataset and the directory of codes according to your need. The .sh files provide the method about how we run the experiments. 
* Prepare Data
 ```bash
python save_data.py
 ```
* Prepare Minimum Absolute Curvature Direction
 ```bash
python min_directions.py
 ```
 * Prepare Normal Vectors
  ```bash
python normal.py
 ```
* No Adversarial Training
 ```bash
python train_only.py -n 1 -g 4 -nr 0
 ```
* Adversarial Training with Projection on Minimum Absolute Curvature Direction
 ```bash
python train_adv.py -n 1 -g 4 -nr 0 --adv 1
 ```
* Adversarial Training with Projection on the Tangent Plane
 ```bash
python train_adv_n.py -n 1 -g 4 -nr 0 --adv 1
 ```
 
 ### Citation
 ```
@inproceedings{wu20223d,
  title={3D point cloud completion with geometric-aware adversarial augmentation},
  author={Wu, Mengxi and Huang, Hao and Fang, Yi},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  pages={4001--4007},
  year={2022},
  organization={IEEE}
}
```
