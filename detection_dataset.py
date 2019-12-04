import cv2
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import pdb
import matplotlib.pyplot as plt
from bounding_box import *

POSITIVE_DATASET_PATH="./datasets/detection/positive"
NEGATIVE_DATASET_PATH="./datasets/detection/negative"
IMAGE_FILE_TYPE = (".jpg", ".png")


class DetectionDataset:
    def __init__(self, hog_converter=None, pos_limit=None, neg_limit=None):
        self.images = {0: [], 1:[]}
        self.hogs = {0: [], 1:[]}
        self.labels = {0: [], 1:[]}
        self._limit = {0: neg_limit, 1: pos_limit}

        self.hog_converter = hog_converter


        self.init_detection_dataset()


    def init_detection_dataset(self):
        self.datasize = {0: 0, 1: 0}
        self.images = {0: [], 1:[]}
        self.hogs = {0: [], 1:[]}
        self.labels = {0: [], 1:[]}
        self._get_detection_dataset(POSITIVE_DATASET_PATH, label=1)
        self._get_detection_dataset(NEGATIVE_DATASET_PATH, label=0)



    def _get_detection_dataset(self, img_path, label=1):
        file_list = listdir(img_path)

        for path_each in file_list:
            if self._limit_reached(label):
                break

            path_curr = join(img_path, path_each)
            if isdir(path_curr):
                self._get_detection_dataset(path_curr, label)
            else:
                self._get_data_from_path(path_curr, label)


    def _get_data_from_path(self, img_path, label):
        if not (img_path.lower().endswith(IMAGE_FILE_TYPE)):
            return

        try:
            img = cv2.imread(img_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = image_resize(img)
            self.images[label].append(img)
            hist = self.hog_converter.compute(img, (8, 8))
            print(hist.shape)
            self.hogs[label].append(hist.reshape(-1, hist.shape[0])[0]) # 15876 hist.shape[0]
            self.labels[label].append(label)
            self.datasize[label] += 1
            print(self.datasize)
            if (self.datasize[label] < 100):
                print(img_path)
        except Exception:
            print("corrupted image detected")

    def _limit_reached(self, label):
        return type(self._limit[label]) != type(None) and self.datasize[label] >= self._limit[label]

def image_resize(img):
    return cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    hog_converter = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    detection_dataset = DetectionDataset(hog_converter)
