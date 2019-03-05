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



e) 训练rpn网络; 训练日志见[train.rpn.log](train.rpn.log)  (optional)

```shell
python train.py --stages rpn
```

f) end2end 方式联合训练rpn和rcnn ;  训练日志见[train.rcnn.log](train.rcnn.log) ; 

```shell
python train.py
```

注：也可以直接联合训练rpn和rcnn；不用训练rpn这一步; 如果没有预先训练rpn;训练时需要训练更多轮。

## 评估

执行如下命令即可

```shell
python evaluate.py
```

输出结果如下：

```
ap:{0: 0.0, 1: 0.5909308613134401, 2: 0.5195146361895532, 3: 0.513946822748238, 4: 0.490973073476093, 5: 0.44759731449012313, 6: 0.43333154556028486, 7: 0.33117353141276584, 8: 0.19636205120076086, 9: 0.3748558681355139, 10: 0.29942392861354306, 11: 0.4040434302434174, 12: 0.611310224080139, 13: 0.4257329355729211, 14: 0.09090909090909091, 15: 0.5597529101905939, 16: 0.33091240504469827, 17: 0.15495867768595042, 18: 0.22335361205793103, 19: 0.5080513656747481, 20: 0.42761708700467765}
mAP:0.39673756858022424

```





## 预测

执行如下命令即可

```shell
python inference.py
```

样例结果如下;更多预测样例见demo_images目录

![examples](demo_images/inferece_examples.2.png)


## todo
0. 评估标准(已完成)
1. 边框裁剪放到Anchors层外面
2. batch_slice部分重构
3. 模型编译重构,分离度量指标
4. GT boxes信息加载部分重构
5. 不同输入尺寸大小精度比较
6. indices类型改为tf.int64;float32类型转换可能丢失精度
