# keras-faster-rcnn

keras 复现论文 [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1504.08083.pdf) ;主要参考了工程[Mask RCNN](https://github.com/matterport/Mask_RCNN); 给出了在Pascal VOC目标检测数据集上的训练和测试过程;

1. [关键点说明](#关键点说明)
2. [依赖](#依赖)
3. [训练](#训练)
4. [预测](#预测)
5. [评估](#评估)
6. [自定义度量](#自定义度量)
7. [toDoList](#toDoList)
8. [总结](#总结)

## 关键点说明:

a.骨干网络使用的是resnet50;Conv1~4用于提取特征;Conv5用于rcnn分类; 精调Conv3~Conv5;所有的Batch Normalization层都不精调，因为batch size太小

b.训练输入图像大小为720\*720; 将图像的长边缩放到720,保持长宽比,短边padding;原文是短边600;长边1000

c.不同于源工程，所有的层都是使用的Keras Layer 包括anchors、rpn网络的target、roi2proposals、detect boxes等

d.为了更好的诊断网络增加了10个自定义度量，如：gt数量，rpn网络匹配的gt数，实际训练的正anchor数等

e.anchor尺寸参考yolo使用聚类得到

f. 原文中rpn网络每张图训练**256**个anchors，正负样本保持1:1；由于PASCAL VOC数据集平均的GT数约为**3**个，rpn网络实际上能够匹配到的正anchors平均约**21**个；所以这里rpn网络每张图训练**80**个anchors

g. 原文中rpn会产生2000个proposals用于rcnn网络训练，但是这里训练几个epoch后，仅仅能够产生不到300个proposals,所有将训练和预测的阶段proposals阶段的iou阈值设置为**0.9**，原文中为**0.7**




## 依赖


tensorflow-gpu 1.14.0



## 训练

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

## 预测

a) 预训练模型下载

​    PASCAL VOC 2007训练集上训练好的模型下载地址： [frcnn-resnet50.035.h5](https://drive.google.com/file/d/1TBGDTpdvCwGhIVEbv4t9q2mpzhL-5HdM/view?usp=sharing)


b) 执行如下命令即可

```shell
python inference.py --weight_path /tmp/frcnn-resnet50.035.h5
```

样例结果如下;更多预测样例见demo_images目录

![examples](demo_images/inferece_examples.2.png)


## 评估

执行如下命令即可

```shell
python evaluate.py --data_set test --weight_path /tmp/frcnn-resnet50.035.h5
```

输出结果如下：

```shell
预测 4952 张图像,耗时：1048.401778936386 秒
ap:{0: 0.0, 1: 0.6768214760548549, 2: 0.6760312377088041, 3: 0.6122568296243334, 4: 0.6367414618873344, 5: 0.6769495102412151, 6: 0.6631589701474099, 7: 0.5463179678366642, 8: 0.3078317216556386, 9: 0.4427729076167924, 10: 0.48129418164837967, 11: 0.5215081042808085, 12: 0.7005935007417052, 13: 0.5352984925242024, 14: 0.3158921077398846, 15: 0.7757831017666008, 16: 0.6199472930246124, 17: 0.32039103004250324, 18: 0.3694302780767011, 19: 0.659186483792924, 20: 0.6523787880767623}
mAP:0.5595292722244065
整个评估过程耗时：1052.8664383888245 秒
```

## 自定义度量


1. gt_num：图像的GT 个数
2. positive_anchor_num：rpn网络实际训练正样本anchor数
3. negative_anchor_num：rpn网络实际训练负样本anchor数
4. rpn_miss_gt_num：rpn网络没有匹配anchor的GT数
5. rpn_gt_min_max_iou：设max_iou为每个GT匹配的anchor最大IoU;rpn_gt_min_max_iou是每个mini-batch中max_iou的最小值;用于衡量anchor长宽尺寸和个数设置是否合理
6. roi_num：经过proposal层nms后实际喂入rcnn网络的proposal个数
7. positive_roi_num：每张图像rcnn网络实际训练的正样本数
8. rcnn_miss_gt_num：rcnn网络按照0.5的iou阈值匹配，有多少个GT没有匹配到proposals；
9. rcnn_miss_gt_num_as：在经过正样本比例(0.25)限制后, 有多少个GT没有匹配到proposals；
10. gt_min_max_iou: 设max_iou为每个GT匹配的proposals最大IoU;gt_min_max_iou是每个mini-batch中max_iou的最小值;用于衡量rpn网络对于边框位置的改善程度                  


## toDoList
0. 评估标准(已完成)
1. 边框裁剪放到Anchors层外面(已完成)
2. batch_slice部分重构(已完成)
3. 模型编译重构,分离度量指标(已完成)
4. GT boxes信息加载部分重构(已完成)
5. 不同输入尺寸大小精度比较
6. indices类型改为tf.int64;float32类型转换可能丢失精度(已完成)
7. 使用聚类GT boxes设计anchors长宽尺寸(已完成)


## 总结
1. 精度还是与原文相差较大，目前在VOC 2007测试集上的mAP为55.9%；过拟合(在训练集上测试78%+)还是BN层没有训练(batch size太小，训练后效果更差)? 
