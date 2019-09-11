'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras import backend as K
K.set_image_data_format('channels_last')

from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length


class CapsNet(tf.keras.Model):
    def __init__(self, shape, n_class=4, **kwargs):
        super(CapsNet, self).__init__(**kwargs)
        self.n_class = n_class
        self.shape = shape
        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')

        # # Reshape layer to be 1 capsule x [filters] atoms
        # _, H, W, C = self.conv1.get_shape()
        self.conv1_reshaped = layers.Reshape((shape[0], shape[1], 1, 16))

        # Layer 1: Primary Capsule: Conv cap with routing 1
        self.primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                        routings=1, name='primarycaps')

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                        routings=3, name='conv_cap_2_1')

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                        routings=3, name='conv_cap_2_2')

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_3_1')

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                        routings=3, name='conv_cap_3_2')

        # Layer 4: Convolutional Capsule
        self.conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_4_1')

        # Layer 1 Up: Deconvolutional Capsule
        self.deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_1_1')

        # Skip connection
        self.up_1 = layers.Concatenate(axis=-2, name='up_1')

        # Layer 1 Up: Deconvolutional Capsule
        self.deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                          padding='same', routings=3, name='deconv_cap_1_2')

        # Layer 2 Up: Deconvolutional Capsule
        self.deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_2_1')

        # Skip connection
        self.up_2 = layers.Concatenate(axis=-2, name='up_2')

        # Layer 2 Up: Deconvolutional Capsule
        self.deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                          padding='same', routings=3, name='deconv_cap_2_2')

        # Layer 3 Up: Deconvolutional Capsule
        self.deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_3_1')

        # Skip connection
        self.up_3 = layers.Concatenate(axis=-2, name='up_3')

        # Layer 4: Convolutional Capsule: 1x1
        self.seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='seg_caps')

        self.reshaper = layers.Reshape((shape[0], shape[1], 16, 1))

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        self.out_seg = layers.Conv2D(filters=n_class, kernel_size=1, name='out_seg', activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        conv1 = self.conv1(inputs)
        conv1_reshaped = self.conv1_reshaped(conv1)
        primary_caps = self.primary_caps(conv1_reshaped)
        conv_cap_2_1 = self.conv_cap_2_1(primary_caps)
        conv_cap_2_2 = self.conv_cap_2_2(conv_cap_2_1)
        conv_cap_3_1 = self.conv_cap_3_1(conv_cap_2_2)
        conv_cap_3_2 = self.conv_cap_3_2(conv_cap_3_1)
        conv_cap_4_1 = self.conv_cap_4_1(conv_cap_3_2)
        deconv_cap_1_1 = self.deconv_cap_1_1(conv_cap_4_1)

        up_1 = self.up_1([deconv_cap_1_1, conv_cap_3_1])
        deconv_cap_1_2 = self.deconv_cap_1_2(up_1)
        deconv_cap_2_1 = self.deconv_cap_2_1(deconv_cap_1_2)

        up_2 = self.up_2([deconv_cap_2_1, conv_cap_2_1])
        deconv_cap_2_2 = self.deconv_cap_2_2(up_2)
        deconv_cap_3_1 = self.deconv_cap_3_1(deconv_cap_2_2)

        up_3 = self.up_3([deconv_cap_3_1, conv1_reshaped])

        seg_caps = self.seg_caps(up_3)
        reshaper = self.reshaper(seg_caps)
        reshaper = tf.squeeze(reshaper, axis=-1)
        out_seg = self.out_seg(reshaper)
        return out_seg

    def summary(self, line_length=None, positions=None, print_fn=None):
        x = tf.keras.Input(shape=(self.shape[0], self.shape[1], self.shape[2]))
        tf.keras.Model(inputs=x, outputs=self.call(x, training=True)).summary(line_length, positions, print_fn)