## Residual Learning for Salient Object Detection.

## Changelog

The code and experimental results have be released now :smile:.

* 2020/8/3:
    * Update README.md. Codes coming soon.

### This is a PyTorch implementation of our TIP 2020. [paper](https://ieeexplore.ieee.org/document/9018384)

## Requirements

- [PyTorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org)

## Notes

The original implementation in this paper is based on [Caffe](https://caffe.berkeleyvision.org/) and applies VGG-16 as backbone. 
This repo is a simple reimplementation of 'ARMI' using PyTorch with ResNet-101 as backbone. Thanks to [PoolNet](https://github.com/backseason/PoolNet) and [DSS-pytorch](https://github.com/AceCoooool/DSS-pytorch)


## Usage 

### 1. Clone the repository

```shell
git clone https://github.com/ArcherFMY/R2Net.git
cd R2Net/
```

### 2. Download the DUTS dataset and unzip it into `data/image_training/` folder.

* [DUTS](http://saliencydetection.net/duts/)
```
|-- data
    |-- image_training
        |-- DUTS-TR-Image
        |-- DUTS-TR-Mask
        |-- train.lst
```

### 3. Train

1. We use ResNet-101 as network backbone and train with a initial lr of 5e-5 for 24 epoches, which is divided by 10 after 15 epochs.
```shell
python main.py --mode='train'
```
2. After training the result model will be stored under `results/run-*`  folder.

### 4. Test

Edit the function `get_test_info` in `main.py` file to add your own dataset for testing. '--sal_mode' indicates different datasets you added.
```shell
python main.py --mode='test' --model='results/run-*/models/final.pth' --test_fold='results/R2Net/ARMI/DUTS-TE/' --sal_mode='duts'
```
The results will be stored under `results/R2Net/ARMI/` folders in `.png` formats. 

### 5. Contact

If you have any questions, feel free to contact me via: `mengyang_feng(at)mail.dlut.edu.cn`

### If you think this work is helpful, please cite
```latex
@article{feng2020residual,
  title={Residual Learning for Salient Object Detection},
  author={Feng, Mengyang and Lu, Huchuan and Yu, Yizhou},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4696--4708},
  year={2020},
  publisher={IEEE}
}
```