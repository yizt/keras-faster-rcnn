# -*- coding: utf-8 -*-
"""
   File Name：     batch_norm.py
   Description :  
   Author :       mick.yi
   Date：          2019/4/26
"""
from tensorflow.python.keras import layers


class BatchNorm(layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        training = False if not self.trainable else training
        return super(self.__class__, self).call(inputs, training=training)
