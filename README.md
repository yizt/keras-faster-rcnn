# keras-faster-rcnn

keras 实现论文 [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1504.08083.pdf) ;主要参考了工程[Mask RCNN](https://github.com/matterport/Mask_RCNN); 给出了在Pascal VOC目标检测数据集上的训练和测试过程;

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

```shell
python train.py
```

f) end2end 方式联合训练rpn和rcnn ;  训练日志见[train.rcnn.log](train.rcnn.log) ; 

```shell
python train.py --stages rcnn
```

注：也可以直接联合训练rpn和rcnn；不用训练rpn这一步; 如果没有预先训练rpn;训练时需要训练更多轮。

## 预测

在jupyter notebook中执行如下命令;详见: [测试.ipynb](测试.ipynb) ;

```
!python inference.py
```

样例结果如下;更多预测样例见demo_images目录

![examples](demo_images/inferece_examples.2.png)



