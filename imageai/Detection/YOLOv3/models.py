from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model, Input



def NetworkConv2D_BN_Leaky(input, channels, kernel_size, kernel_regularizer = l2(5e-4), strides=(1,1), padding="same", use_bias=False):

    network = Conv2D( filters=channels, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=kernel_regularizer, use_bias=use_bias)(input)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=0.1)(network)
    return network

def residual_block(input, channels, num_blocks):
    network = ZeroPadding2D(((1,0), (1,0)))(input)
    network = NetworkConv2D_BN_Leaky(input=network,channels=channels, kernel_size=(3,3), strides=(2,2), padding="valid")

    for blocks in range(num_blocks):
        network_1 = NetworkConv2D_BN_Leaky(input=network, channels= channels // 2, kernel_size=(1,1))
        network_1 = NetworkConv2D_BN_Leaky(input=network_1,channels= channels, kernel_size=(3,3))

        network = Add()([network, network_1])
    return network

def darknet(input):
    network = NetworkConv2D_BN_Leaky(input=input, channels=32, kernel_size=(3,3))
    network = residual_block(input=network, channels=64, num_blocks=1)
    network = residual_block(input=network, channels=128, num_blocks=2)
    network = residual_block(input=network, channels=256, num_blocks=8)
    network = residual_block(input=network, channels=512, num_blocks=8)
    network = residual_block(input=network, channels=1024, num_blocks=4)


    return network

def last_layers(input, channels_in, channels_out, layer_name=""):



    network = NetworkConv2D_BN_Leaky( input=input, channels=channels_in, kernel_size=(1,1))
    network = NetworkConv2D_BN_Leaky(input=network, channels= (channels_in * 2) , kernel_size=(3, 3))
    network = NetworkConv2D_BN_Leaky(input=network, channels=channels_in, kernel_size=(1, 1))
    network = NetworkConv2D_BN_Leaky(input=network, channels=(channels_in * 2), kernel_size=(3, 3))
    network = NetworkConv2D_BN_Leaky(input=network, channels=channels_in, kernel_size=(1, 1))

    network_1 = NetworkConv2D_BN_Leaky(input=network, channels=(channels_in * 2), kernel_size=(3, 3))
    network_1 = Conv2D(filters=channels_out, kernel_size=(1,1), name=layer_name)(network_1)

    return  network, network_1

def yolo_main(input, num_anchors, num_classes):

    darknet_network = Model(input, darknet(input))

    network, network_1 = last_layers(darknet_network.output, 512, num_anchors * (num_classes + 5), layer_name="last1")

    network = NetworkConv2D_BN_Leaky( input=network, channels=256, kernel_size=(1,1))
    network = UpSampling2D(2)(network)
    network = Concatenate()([network, darknet_network.layers[152].output])

    network, network_2 = last_layers(network,  256,  num_anchors * (num_classes + 5), layer_name="last2")

    network = NetworkConv2D_BN_Leaky(input=network, channels=128, kernel_size=(1, 1))
    network = UpSampling2D(2)(network)
    network = Concatenate()([network, darknet_network.layers[92].output])

    network, network_3 = last_layers(network, 128, num_anchors * (num_classes + 5), layer_name="last3")

    return Model(input, [network_1, network_2, network_3])


def tiny_yolo_main(input, num_anchors, num_classes):
    network_1 = NetworkConv2D_BN_Leaky(input=input, channels=16, kernel_size=(3,3) )
    network_1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=32, kernel_size=(3, 3))
    network_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=64, kernel_size=(3, 3))
    network_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=128, kernel_size=(3, 3))
    network_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_1 = NetworkConv2D_BN_Leaky(input=network_1, channels=256, kernel_size=(3, 3))

    network_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network_1)
    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=512, kernel_size=(3, 3))
    network_2 = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same")(network_2)
    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=1024, kernel_size=(3, 3))
    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=256, kernel_size=(1, 1))

    network_3 = NetworkConv2D_BN_Leaky(input=network_2, channels=512, kernel_size=(3, 3))
    network_3 = Conv2D(num_anchors * (num_classes + 5),  kernel_size=(1,1))(network_3)

    network_2 = NetworkConv2D_BN_Leaky(input=network_2, channels=128, kernel_size=(1, 1))
    network_2 = UpSampling2D(2)(network_2)

    network_4 = Concatenate()([network_2, network_1])
    network_4 = NetworkConv2D_BN_Leaky(input=network_4, channels=256, kernel_size=(3, 3))
    network_4 = Conv2D(num_anchors * (num_classes + 5), kernel_size=(1,1))(network_4)

    return Model(input, [network_3, network_4])