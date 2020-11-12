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
train, train_view_pos = DataLoader.img_to_np(path_train, size)
print("test")
test, test_view_pos = DataLoader.img_to_np(path_test, size)

train_pa = train[train_view_pos == "PA"]
train_ap = train[train_view_pos == "AP"]

test_pa = test[test_view_pos == "PA"]
test_ap = test[test_view_pos == "AP"]

model_pa = Model(train_pa)
model_pa.fit()

pred_pa = model_pa.predict(test_pa)
print(list(pred_pa['data'].keys()))

target = np.zeros(test.shape[0],).astype(int)  # all normal CIFAR10 training instances
labels = ['normal', 'outlier']
plot_instance_score(pred_pa, target, labels, model_pa.get_threshold())


model_ap = Model(train_ap)
model_ap.fit()

pred_ap = model_ap.predict(test_ap)
print(list(pred_ap['data'].keys()))

target = np.zeros(test.shape[0],).astype(int)  # all normal CIFAR10 training instances
labels = ['normal', 'outlier']
plot_instance_score(pred_ap, target, labels, model_ap.get_threshold())