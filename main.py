import os
import numpy as np
import pandas as pd
from alibi_detect.utils.visualize import plot_instance_score
from data_loader import DataLoader
from model import Model

is_kaggle = False if 'rafael' in os.getcwd() else True

size = (64, 64)

if is_kaggle:
    path_train = "../input/rsna-pneumonia-detection-challenge/stage_2_train_images"
    path_test = "../input/rsna-pneumonia-detection-challenge/stage_2_test_images"

    train_df = pd.read_csv("../input/chest-xray-anomaly-detection/train.csv")
    path_train_sub = "../input/chest-xray-anomaly-detection/images"

    sub_df = pd.read_csv("../input/chest-xray-anomaly-detection/sample_submission.csv")
    path_test_sub = "../input/chest-xray-anomaly-detection/images"
else:
    path_train = "/home/rafael/repo/AnomalyDetection/Data/stage_2_train_images"
    path_test = "/home/rafael/repo/AnomalyDetection/Data/stage_2_test_images"

    train_df = pd.read_csv("/home/rafael/repo/AnomalyDetection/Data/train.csv")
    path_train_sub = "/home/rafael/repo/AnomalyDetection/Data/images"

    sub_df = pd.read_csv("/home/rafael/repo/AnomalyDetection/Data/sample_submission.csv")
    path_test_sub = "/home/rafael/repo/AnomalyDetection/Data/images"

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
target_ap = np.zeros(test_ap.shape[0], ).astype(int)  # all normal CIFAR10 training instances
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


# SUB
train_sub = DataLoader.imgs_to_np(file_names=train_df["fileName"].tolist(),
                                  size=size,
                                  path=path_train_sub)
modality = ["Not Anomaly", "Anomaly"]
target_train = train_df["anomaly"].to_numpy()
print("PA ON TRAIN")
train_pa = model_pa.predict(train_sub)
plot_instance_score(train_pa, target_train, modality, model_pa.get_threshold())

print("AP ON TRAIN")
subs_ap = model_ap.predict(train_sub)
plot_instance_score(subs_ap, target_train, modality, model_ap.get_threshold())

test_sub = DataLoader.imgs_to_np(file_names=sub_df["fileName"].tolist(),
                                 size=size,
                                 path=path_test_sub)
target_subs = np.zeros(test_sub.shape[0], ).astype(int)  # all normal CIFAR10 training instances
print("PA ON TEST")
subs_pa = model_pa.predict(test_sub)
plot_instance_score(subs_pa, target_subs, labels, model_pa.get_threshold())

# AP ON AP
print("AP ON TEST")
subs_ap = model_ap.predict(test_sub)
plot_instance_score(subs_ap, target_subs, labels, model_ap.get_threshold())
