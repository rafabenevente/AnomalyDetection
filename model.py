import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    Dense, Reshape, InputLayer, Flatten
from alibi_detect.od import OutlierAE


class Model:
    def __init__(self, data, shape):
        encoding_dim = 1024
        dense_dim = [8, 8, 128]

        self.train_data = data
        self.shape = shape

        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=self.shape),
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

        self.enc = OutlierAE(threshold=0.001,
                             encoder_net=encoder_net,
                             decoder_net=decoder_net)

    def fit(self, verbose):
        adam = tf.keras.optimizers.Adam(lr=1e-4)

        self.enc.fit(self.train_data, epochs=1000, verbose=verbose,
                     optimizer=adam)

    def predict(self, test_data):
        return self.enc.predict(test_data,
                                outlier_type='instance',  # use 'feature' or 'instance' level
                                return_feature_score=True,  # scores used to determine outliers
                                return_instance_score=True)

    def get_threshold(self):
        return self.enc.threshold
