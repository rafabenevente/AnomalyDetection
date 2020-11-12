import os
import cv2
import numpy as np
import pydicom

class DataLoader(object):
    @staticmethod
    def img_to_np(path, size, resize=True):
        img_array = []
        view_pos = []
        for fname in os.listdir(path):
            dicom = pydicom.dcmread(os.path.join(path, fname))
            img = dicom.pixel_array
            if (resize):
                img = cv2.resize(img, size)
            img = img.astype('float32') / 255.
            img = np.stack((img,) * 3, axis=-1)
            img_array.append(np.asarray(img))
            view_pos.append(dicom.ViewPosition)
        img_array = np.array(img_array)
        view_pos = np.array(view_pos)
        return img_array, view_pos
