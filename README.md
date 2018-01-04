# Semantic Segmentation with MobileNets and PSP Module

## Setup
Environment Details:
The code has been tested with Ubuntu 16.04 LTS, Intel i5-3570, 4 Cores @ 3.40 GHz, NVIDIA GeForce GTX 1080

First, install NVIDIA drivers and check whether they are working with `nvidia-smi`.

-  CUDA 8.0.61 (https://developer.nvidia.com/cuda-80-ga2-download-archive)
    - Install both (Base Installer and Patch 2) and refuse wherever asked to overwrite NVIDIA drivers.

- cuDNN v5.1
    - Download cudnn-8.0-linux-x64-v6.0.tgz and do the following.
        ```shell
        tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
        cp cuda/lib64/* /usr/local/cuda/lib64/
        cp cuda/include/cudnn.h /usr/local/cuda/include/
        ```

- Tensorflow 1.3
    - To install TF 1.3
        ```shell
        pip install tensorflow-gpu==1.3
        ```

## Usage

#### Training

1. The weights for the model can be initialized by using pretrained weights of [MobileNet](https://github.com/Zehaos/MobileNet) and initializing rest of weights of PSP Module from [PSPNet](https://github.com/hszhao/PSPNet) (conv weights with Xavier initializer & biases initialized as 0).
<!-- ```
python prepare_initialisation_weights.py --pretrained_mobilenet=MobileNetPreTrained/model.ckpt-906808 --save_model=MobileNetPSP
``` -->
   Weights have already been saved in **`MobileNetPSP`**, so this part can be skipped.

2. The dataset can be downloaded from [here](https://www.cityscapes-dataset.com/downloads/). The dataset should be kept in the directory structure as shown below:
```
cityscapes-images
├── leftImg8bit
├── gtCoarse
└── gtFine
```
**`gtFine`** and **`gtCoarse`** should contain the **`*_gtFine_labelTrainIds.png`** which are the ground truth labels generated by **`createTrainIdLabelImgs.py`** in [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).

List with images and ground truth paths can be generated using **`generate_image_list.py`**. This step has been done and different lists are saved in **`list`**.

3. Train the model using
```
python train.py --data_dir=PATH_TO_cityscapes-images_FOLDER --log_dir=logs/train1 --num_epochs=80 --gpu=0 --update_beta=True --update_mean_var=True
```
Train the model for some time. Then, change `update_mean_var` to False and train for more time. Finally, train with `--update_beta=False --update_mean_var=False` for rest of the time.

**`train_continue.py`** can be used to start the training from certain epoch if you wish to stop the training and change parameters. For example:
```
python train_continue.py --pretrained_checkpoint=logs/train1/checkpoints/model.ckpt --log_dir=logs/train1 --num_epochs=80 --start_epoch=81 --gpu=0 --update_mean_var=False --update_beta=True
```

Training Procedure on Fine Dataset:
```
python train.py --log_dir=logs/train1 --gpu=0 --num_epochs=80 --optimizer=momentum --update_mean_var=True --update_beta=True
python train_continue.py --pretrained_checkpoint=logs/train1/checkpoints/model.ckpt-80 --log_dir=logs/train1 --gpu=0 --num_epochs=80 --start_epoch=81 --update_mean_var=False --update_beta=True
python train_continue.py --pretrained_checkpoint=logs/train1/checkpoints/model.ckpt-160 --log_dir=logs/train1 --gpu=0 --num_epochs=100 --start_epoch=161 --update_mean_var=False --update_beta=False
```

Training procedures can also be found in **`train.sh`**. It can be used by changing `DATASET_DIR`.
```
chmod +x train.sh
./train.sh
```

##### Training on Coarse dataset
For training on Coarse Dataset, `--data_list=list/train_list.txt` needs to be changed to `--data_list=list/train_extra_list.txt`

Its training procedure can be found in **`train_coarse.sh`**.

#### Evaluation
Models can be evaluated by
```
python evaluate.py --checkpoint_path=logs/train1/model.ckpt
```

`evaluate_model.sh` can be used to evaluate a model by changing its variables: `DATASET_DIR`, `DATA_LIST`, `CHECKPOINT_FOLDER`, `EPOCH`.

`evaluate_models_it.sh` can be used to evaluate all models from range `START` to `END`.  
`evaluate_models_list.sh` can be used to evaluate models in the list `EPOCHS`.

## Results


After training on [Fine annotated](https://www.cityscapes-dataset.com/examples/#fine-annotations) Cityscapes dataset, evaluation on Cityscapes validation dataset gives **61% mIoU** without Flipping.  
Inference Time: **52ms** on GPU (TF 1.3), **3.34s** on CPU (this is wrong!)  
Trained Weights Size: **69MB**  
Few results are shown.

| Input Image | Prediction | Ground Truth |
|--------|:---------:|:---------:|
| ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/1_im.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/1_pred.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/1_gt.png) |
| ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/4_im.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/4_pred.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/4_gt.png) |
| ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/6_im.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/6_pred.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/6_gt.png) |
| ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/7_im.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/7_pred.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/7_gt.png) |
| ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/8_im.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/8_pred.png) | ![](https://github.com/interritus1996/mobilenet_psp/blob/master/results/8_gt.png) |
<!-- | ![]() | ![]() | ![]() | -->


## TODO
- [ ] Train on Coarse Dataset and report results.
- [ ] Add Auxilary loss.



## References

1. **Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam.** _MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications_, 2017 [[arxiv][1]]
1. **Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia.** _Pyramid Scene Parsing Network_, 2017 [[arxiv][2]]

[1]: https://arxiv.org/abs/1704.04861
[2]: https://arxiv.org/abs/1612.01105
