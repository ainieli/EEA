# Data Augmentation with Multi-armed Bandit on Image Deformations Improves Fluorescence Glioma Boundary Recognition

---

This is the official implementation of **MICCAI 2024** paper "***Data Augmentation with Multi-armed Bandit on Image Deformations Improves Fluorescence Glioma Boundary Recognition***". 

The proposed data augmentation method "Explore and Exploit Augment" (EEA) can improve the representation ability of small-scale data to improve the classification performance of different types of DL models.

<p align="center">
    <img src="Fig.1.jpg" alt="EEA" width="80%">
</p>

---

## Start

- pytorch 1.13.1

- timm 0.6.7

- numpy 1.21.5

## Training

1. Create your own dataloader and DL classification model

2. Change settings in ”train.py“

3. Run "train.py"

## Cite this paper

```
@inproceedings{xiao2024data,
  title={Data Augmentation with Multi-armed Bandit on Image Deformations Improves Fluorescence Glioma Boundary Recognition},
  author={Xiao, Anqi and Han, Keyi and Shi, Xiaojing and Tian, Jie and Hu, Zhenhua},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={130--140},
  year={2024},
  organization={Springer}
}
```
