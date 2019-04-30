# keras-faster-rcnn

keras 复现论文 [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1504.08083.pdf) ;主要参考了工程[Mask RCNN](https://github.com/matterport/Mask_RCNN); 给出了在Pascal VOC目标检测数据集上的训练和测试过程;

**关键点说明**:

a.骨干网络使用的是resnet50

b.训练输入图像大小为720*720; 将图像的长边缩放到720,保持长宽比,短边padding;原文是短边600;长边1000

c.不同于源工程，所有的层都是使用的Keras Layer 包括anchors、rpn网络的target、roi2proposals等

d.为了更好的诊断网络增加了10个自定义度量，如：gt数量，rpn网络匹配的gt数，实际训练的正样本数等



[TOC]

## 依赖

Keras 2.2.4

tensorflow-gpu 1.9.0


## 预测



执行如下命令即可

```shell
python inference.py
```



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

e) 聚类生成anchors长宽
```shell
python gt_cluster.py --clusters 9
```
   结果如下(每次聚类结果有细微差异):
```shell
h:[53.25, 83.97, 149.73, 187.92, 202.07, 270.18, 357.79, 445.08, 488.2] 
w:[40.64, 128.43, 67.18, 304.62, 141.64, 539.33, 188.27, 336.6, 588.34]
ratio:[1.31, 0.65, 2.23, 0.62, 1.43, 0.5, 1.9, 1.32, 0.83]
ious:{1: 0.68557111415695, 2: 0.7254386539924431, 3: 0.692596649433563, 4: 0.6834624200806161, 5: 0.6157859655510957, 6: 0.6595932146657101, 7: 0.6685312631536455, 8: 0.665801489785847, 9: 0.6669742441576416, 10: 0.6121460291384846, 11: 0.6710420495151501, 12: 0.7141121719703986, 13: 0.720584796915154, 14: 0.5639737876945834, 15: 0.7321887018308443, 16: 0.6533920906006669, 17: 0.6594010218140989, 18: 0.5846891288412122, 19: 0.7055301287233066, 20: 0.6066710334483025}
mean iou:0.6643742977734857
```
   并根据聚类结果设置config.py中的，anchors长宽属性
```python
    RPN_ANCHOR_HEIGHTS = [52.42, 85.64, 143.89, 186.92, 208.26, 266.1, 359.72, 446.26, 484.92]
    RPN_ANCHOR_WIDTHS = [40.85, 132.84, 66.24, 294.26, 135.53, 533.3, 190.26, 339.55, 591.88]
```
f) end2end方式训练网络
```shell
python train.py
```

## 评估

执行如下命令即可

```shell
python evaluate.py --data_set test --weight_path /tmp/frcnn-resnet50.100.h5
```

输出结果如下：

```shell
预测 4952 张图像,耗时：822.7565138339996 秒
ap:{0: 0.0, 1: 0.6590661667560703, 2: 0.5958548790033925, 3: 0.638358541906528, 4: 0.6107938445652822, 5: 0.6772344340797694, 6: 0.6608167190974154, 7: 0.49887556111945863, 8: 0.30730381054734174, 9: 0.47056113146487555, 10: 0.43225705232604256, 11: 0.5153157639863547, 12: 0.6845281416752329, 13: 0.5396385189363719, 14: 0.2967691770090697, 15: 0.695371532493559, 16: 0.5161533797396031, 17: 0.3179253341562169, 18: 0.33037834953169576, 19: 0.633266549284698, 20: 0.5763473043533416}
mAP:0.532840809601616
整个评估过程耗时：826.3198778629303 秒
```


样例结果如下;更多预测样例见demo_images目录

![examples](demo_images/inferece_examples.2.png)


## toDoList
0. 评估标准(已完成)
1. 边框裁剪放到Anchors层外面(已完成)
2. batch_slice部分重构(已完成)
3. 模型编译重构,分离度量指标(已完成)
4. GT boxes信息加载部分重构(已完成)
5. 不同输入尺寸大小精度比较
6. indices类型改为tf.int64;float32类型转换可能丢失精度(已完成)


## 总结
1. 裁剪到输入比裁剪到窗口效果好
