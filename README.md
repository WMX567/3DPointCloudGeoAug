## 3D Point Cloud Completion with Geometric-Aware Adversarial Augmentation
This repository contains the implementation for the paper "Point Cloud Completion with Geometric-Aware Adversarial Augmentation", ICPR 2022.

### Models

Point Completion Network [arXiv](https://arxiv.org/pdf/1808.00671.pdf)

### Dependencies
* Python 3.7.9
* PyTorch 1.7.0
* CUDA 10.1.243
* Open3d
* pyemd

### Dataset
ShapeNet [link](https://shapenet.org/)
To prepare the data, please use following commands
```bash
python save
 ```

### Run Experiments
* Prepare Data
 ```bash
python save_data.py
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
@article{wu20213d,
  title={3D Point Cloud Completion with Geometric-Aware Adversarial Augmentation},
  author={Wu, Mengxi and Huang, Hao and Fang, Yi},
  journal={arXiv preprint arXiv:2109.10161},
  year={2021}
}
```
