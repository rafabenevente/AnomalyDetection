import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,\
    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE

class Model:
    @staticmethod
    def get_model(shape):
        encoding_dim = 1024
        dense_dim = [8, 8, 128]

        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=shape),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(encoding_dim, )
            ])

        decoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(encoding_dim,)),
                Dense(np.prod(dense_dim)),
                Reshape(target_shape=dense_dim),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
            ])

        od = OutlierAE(threshold=0.001,
                       encoder_net=encoder_net,
                       decoder_net=decoder_net)

        return od