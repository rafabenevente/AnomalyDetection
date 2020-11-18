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
            if resize:
                img = cv2.resize(img, size)
            img = img.astype('float32') / 255.
            img = np.stack((img,) * 3, axis=-1)
            img_array.append(np.asarray(img))
            view_pos.append(dicom.ViewPosition)
        img_array = np.array(img_array)
        view_pos = np.array(view_pos)
        return img_array, view_pos

    @staticmethod
    def dicom_to_file(path, output_path, size, resize=True):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            os.mkdir(os.path.join(output_path, "PA"))
            os.mkdir(os.path.join(output_path, "AP"))

        for fname in os.listdir(path):
            dicom = pydicom.dcmread(os.path.join(path, fname))
            img = dicom.pixel_array
            if resize:
                img = cv2.resize(img, size)
            img = img.astype('float32') / 255.
            img = np.stack((img,) * 3, axis=-1)
            cv2.imwrite(os.path.join(output_path,
                                     dicom.ViewPosition,
                                     fname.split(".")[0]+".png"),
                        img)

    @staticmethod
    def imgs_to_np(file_names, path, size, resize=True):
        img_array = []
        for fname in file_names:
            img = cv2.imread(os.path.join(path, fname))
            if resize:
                img = cv2.resize(img, size)
            img = img.astype('float32') / 255.
            img_array.append(img)
        img_array = np.array(img_array)
        return img_array
