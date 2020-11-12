import os
import numpy as np
from alibi_detect.utils.visualize import plot_instance_score
from data_loader import DataLoader
from model import Model

size = (64, 64)

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

model_pa = Model(data=train_pa)
model_pa.fit(verbose=False)

model_ap = Model(data=train_ap)
model_ap.fit(verbose=False)

target_pa = np.zeros(test_pa.shape[0], ).astype(int)  # all normal CIFAR10 training instances
target_ap = np.zeros(test_pa.shape[0], ).astype(int)  # all normal CIFAR10 training instances
labels = ['normal', 'outlier']

# PA ON PA
print("PA ON PA")
pred_pa_pa = model_pa.predict(test_pa)
plot_instance_score(pred_pa_pa, target_pa, labels, model_pa.get_threshold())

# PA ON AP
print("PA ON AP")
pred_pa_ap = model_pa.predict(test_ap)
plot_instance_score(pred_pa_ap, target_ap, labels, model_pa.get_threshold())

# AP ON AP
print("AP ON AP")
pred_ap_ap = model_ap.predict(test_ap)
plot_instance_score(pred_ap_ap, target_ap, labels, model_ap.get_threshold())

# AP ON PA
print("AP ON PA")
pred_ap_pa = model_ap.predict(test_pa)
plot_instance_score(pred_ap_pa, target_pa, labels, model_ap.get_threshold())
