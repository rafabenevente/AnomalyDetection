import os
import tensorflow as tf
import numpy as np
from alibi_detect.utils.visualize import plot_instance_score
from data_loader import DataLoader
from model import Model

size = (64,64)

if 'rafael' in os.getcwd():
    path_train = "/home/rafael/repo/AnomalyDetection/Data/stage_2_train_images"
    path_test = "/home/rafael/repo/AnomalyDetection/Data/stage_2_test_images"
else:
    path_train = "../input/rsna-pneumonia-detection-challenge/stage_2_train_images"
    path_test = "../input/rsna-pneumonia-detection-challenge/stage_2_test_images"

print("train")
train = DataLoader.img_to_np(path_train, size)
print("test")
test = DataLoader.img_to_np(path_test, size)

enc = Model.get_model(train[0].shape)

adam = tf.keras.optimizers.Adam(lr=1e-4)

enc.fit(train, epochs=100, verbose=True,
       optimizer = adam)

# enc.infer_threshold(test, threshold_perc=95)
#
# preds = enc.predict(test, outlier_type='instance',
#             return_instance_score=True,
#             return_feature_score=True)

od_preds = enc.predict(test,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)
print(list(od_preds['data'].keys()))

target = np.zeros(test.shape[0],).astype(int)  # all normal CIFAR10 training instances
labels = ['normal', 'outlier']
plot_instance_score(od_preds, target, labels, enc.threshold)