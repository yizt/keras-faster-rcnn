# keras-faster-rcnn

keras 实现论文 [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1504.08083.pdf) ;主要参考了工程[Mask RCNN](https://github.com/matterport/Mask_RCNN); 给出了在Pascal VOC目标检测数据集上的训练和测试过程

[TOC]

## 依赖

Keras 2.2.4

tensorflow-gpu 1.9.0



## 训练网络

a) 下载工程

```shell
git clone https://github.com/yizt/keras-faster-rcnn.git
```

b) 下载pascal voc数据集,并解压

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

c) 下载resnet 50预训练模型

```shell
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
```

d) 修改工程faster_rcnn/config.py文件中数据集和预训练模型路径

```python
pretrained_weights = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
voc_path = '/opt/dataset/VOCdevkit'
```



e) 训练rpn网络; 训练日志见[train.rpn.log](train.rpn.log)

```python
python train.py
```

f) end2end 方式联合训练rpn和rcnn 

```python
python train.py --stages rcnn
```



## 预测

