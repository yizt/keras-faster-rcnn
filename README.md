# keras-faster-rcnn

## 训练
python train.py
python train.py --stages rcnn

## rpn 训练日志
==================================================================================================
Epoch 1/10
4981/4981 [==============================] - 338s 68ms/step - loss: 2.5112 - rpn_bbox_loss: 1.9086 - rpn_class_loss: 0.6026 - gt_num: 3.0464 - positive_anchor_num: 67.8606 - miss_match_gt_num: 1.2468

Epoch 00001: saving model to /tmp/frcnn-rpn.001.h5
Epoch 2/10
4981/4981 [==============================] - 331s 66ms/step - loss: 1.8275 - rpn_bbox_loss: 1.3767 - rpn_class_loss: 0.4508 - gt_num: 3.1007 - positive_anchor_num: 67.7074 - miss_match_gt_num: 1.2988

Epoch 00002: saving model to /tmp/frcnn-rpn.002.h5
Epoch 3/10
4981/4981 [==============================] - 331s 67ms/step - loss: 1.4118 - rpn_bbox_loss: 1.0222 - rpn_class_loss: 0.3896 - gt_num: 3.0336 - positive_anchor_num: 68.9197 - miss_match_gt_num: 1.2297

Epoch 00003: saving model to /tmp/frcnn-rpn.003.h5
Epoch 4/10
4981/4981 [==============================] - 332s 67ms/step - loss: 1.1416 - rpn_bbox_loss: 0.7822 - rpn_class_loss: 0.3594 - gt_num: 3.1377 - positive_anchor_num: 68.7522 - miss_match_gt_num: 1.3090

Epoch 00004: saving model to /tmp/frcnn-rpn.004.h5
Epoch 5/10
4981/4981 [==============================] - 332s 67ms/step - loss: 0.9669 - rpn_bbox_loss: 0.6315 - rpn_class_loss: 0.3354 - gt_num: 3.0458 - positive_anchor_num: 67.8435 - miss_match_gt_num: 1.2309

Epoch 00005: saving model to /tmp/frcnn-rpn.005.h5
Epoch 6/10
4981/4981 [==============================] - 332s 67ms/step - loss: 0.8656 - rpn_bbox_loss: 0.5451 - rpn_class_loss: 0.3205 - gt_num: 3.1517 - positive_anchor_num: 67.5460 - miss_match_gt_num: 1.3456

Epoch 00006: saving model to /tmp/frcnn-rpn.006.h5
Epoch 7/10
4981/4981 [==============================] - 329s 66ms/step - loss: 0.8011 - rpn_bbox_loss: 0.4941 - rpn_class_loss: 0.3070 - gt_num: 3.0512 - positive_anchor_num: 67.8810 - miss_match_gt_num: 1.2590

Epoch 00007: saving model to /tmp/frcnn-rpn.007.h5
Epoch 8/10
4981/4981 [==============================] - 329s 66ms/step - loss: 0.7572 - rpn_bbox_loss: 0.4615 - rpn_class_loss: 0.2957 - gt_num: 3.0657 - positive_anchor_num: 67.6639 - miss_match_gt_num: 1.2859

Epoch 00008: saving model to /tmp/frcnn-rpn.008.h5
Epoch 9/10
4981/4981 [==============================] - 330s 66ms/step - loss: 0.7369 - rpn_bbox_loss: 0.4454 - rpn_class_loss: 0.2916 - gt_num: 3.0966 - positive_anchor_num: 68.3839 - miss_match_gt_num: 1.2938

Epoch 00009: saving model to /tmp/frcnn-rpn.009.h5
Epoch 10/10
4981/4981 [==============================] - 330s 66ms/step - loss: 0.7169 - rpn_bbox_loss: 0.4297 - rpn_class_loss: 0.2872 - gt_num: 3.1057 - positive_anchor_num: 67.8729 - miss_match_gt_num: 1.3021

Epoch 00010: saving model to /tmp/frcnn-rpn.010.h5



## frcnn 网络训练日志
Epoch 1/10
4981/4981 [==============================] - 1597s 321ms/step - loss: 2.3410 - rpn_bbox_loss: 0.4264 - rpn_class_loss: 0.2826 - rcnn_bbox_loss: 0.9366 - rcnn_class_loss: 0.6955 - gt_num: 3.0882 - positive_anchor_num: 68.1956 - miss_match_gt_num: 1.2772 - rcnn_miss_match_gt_num: 1.4204      

Epoch 00001: saving model to /tmp/frcnn-rcnn.001.h5
Epoch 2/10
4981/4981 [==============================] - 1598s 321ms/step - loss: 1.9258 - rpn_bbox_loss: 0.4325 - rpn_class_loss: 0.2803 - rcnn_bbox_loss: 0.8374 - rcnn_class_loss: 0.3755 - gt_num: 3.0783 - positive_anchor_num: 67.7406 - miss_match_gt_num: 1.2786 - rcnn_miss_match_gt_num: 1.4528  

Epoch 00002: saving model to /tmp/frcnn-rcnn.002.h5
Epoch 3/10
4981/4981 [==============================] - 1598s 321ms/step - loss: 1.8605 - rpn_bbox_loss: 0.4324 - rpn_class_loss: 0.2774 - rcnn_bbox_loss: 0.7997 - rcnn_class_loss: 0.3509 - gt_num: 3.1018 - positive_anchor_num: 67.0151 - miss_match_gt_num: 1.3016 - rcnn_miss_match_gt_num: 1.4806

Epoch 00003: saving model to /tmp/frcnn-rcnn.003.h5
Epoch 4/10
4981/4981 [==============================] - 1592s 320ms/step - loss: 1.8140 - rpn_bbox_loss: 0.4321 - rpn_class_loss: 0.2773 - rcnn_bbox_loss: 0.7606 - rcnn_class_loss: 0.3440 - gt_num: 3.0900 - positive_anchor_num: 67.9884 - miss_match_gt_num: 1.2792 - rcnn_miss_match_gt_num: 1.4731

Epoch 00004: saving model to /tmp/frcnn-rcnn.004.h5
Epoch 5/10
4981/4981 [==============================] - 1586s 318ms/step - loss: 1.7934 - rpn_bbox_loss: 0.4318 - rpn_class_loss: 0.2751 - rcnn_bbox_loss: 0.7470 - rcnn_class_loss: 0.3395 - gt_num: 3.0948 - positive_anchor_num: 67.3398 - miss_match_gt_num: 1.2958 - rcnn_miss_match_gt_num: 1.4937

Epoch 00005: saving model to /tmp/frcnn-rcnn.005.h5
Epoch 6/10
4981/4981 [==============================] - 1584s 318ms/step - loss: 1.7612 - rpn_bbox_loss: 0.4291 - rpn_class_loss: 0.2719 - rcnn_bbox_loss: 0.7189 - rcnn_class_loss: 0.3414 - gt_num: 3.0089 - positive_anchor_num: 68.0191 - miss_match_gt_num: 1.2223 - rcnn_miss_match_gt_num: 1.4176

Epoch 00006: saving model to /tmp/frcnn-rcnn.006.h5
Epoch 7/10
4981/4981 [==============================] - 1585s 318ms/step - loss: 1.7366 - rpn_bbox_loss: 0.4287 - rpn_class_loss: 0.2702 - rcnn_bbox_loss: 0.6935 - rcnn_class_loss: 0.3441 - gt_num: 3.0301 - positive_anchor_num: 68.7414 - miss_match_gt_num: 1.2282 - rcnn_miss_match_gt_num: 1.4343

Epoch 00007: saving model to /tmp/frcnn-rcnn.007.h5
Epoch 8/10
4981/4981 [==============================] - 1583s 318ms/step - loss: 1.7301 - rpn_bbox_loss: 0.4288 - rpn_class_loss: 0.2705 - rcnn_bbox_loss: 0.6875 - rcnn_class_loss: 0.3433 - gt_num: 3.0674 - positive_anchor_num: 67.8348 - miss_match_gt_num: 1.2727 - rcnn_miss_match_gt_num: 1.4926  

Epoch 00008: saving model to /tmp/frcnn-rcnn.008.h5
Epoch 9/10
4981/4981 [==============================] - 1583s 318ms/step - loss: 1.7189 - rpn_bbox_loss: 0.4258 - rpn_class_loss: 0.2668 - rcnn_bbox_loss: 0.6824 - rcnn_class_loss: 0.3439 - gt_num: 3.0566 - positive_anchor_num: 67.5226 - miss_match_gt_num: 1.2568 - rcnn_miss_match_gt_num: 1.4815

Epoch 00009: saving model to /tmp/frcnn-rcnn.009.h5
Epoch 10/10
4981/4981 [==============================] - 1585s 318ms/step - loss: 1.7007 - rpn_bbox_loss: 0.4208 - rpn_class_loss: 0.2664 - rcnn_bbox_loss: 0.6702 - rcnn_class_loss: 0.3432 - gt_num: 3.0775 - positive_anchor_num: 67.6362 - miss_match_gt_num: 1.2760 - rcnn_miss_match_gt_num: 1.5035

Epoch 00010: saving model to /tmp/frcnn-rcnn.010.h5

