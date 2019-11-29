import cv2
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import pdb


POSITIVE_DATASET_PATH="./datasets/detection/positive"
NEGATIVE_DATASET_PATH="./datasets/detection/negative"
IMAGE_FILE_TYPE = (".jpg", ".png")


class DetectionDataset:
    def __init__(self, hog_converter=None, limit=None):
        self.images = []
        self.hogs = []
        self.labels = []
        self._limit = limit

        self.hog_converter = hog_converter


        self.init_detection_dataset()


    def init_detection_dataset(self):
        self.datasize = 0
        self.images = []
        self.hogs = []
        self.labels = []
        self._get_detection_dataset(POSITIVE_DATASET_PATH, label=1)
        self._get_detection_dataset(NEGATIVE_DATASET_PATH, label=0)



    def _get_detection_dataset(self, img_path, label=1):
        file_list = listdir(img_path)

        for path_each in file_list:
            if self._limit_reached():
                break

            path_curr = join(img_path, path_each)
            if isdir(path_curr):
                self._get_detection_dataset(path_curr, label)
            else:
                self._get_data_from_path(path_curr, label)


    def _get_data_from_path(self, img_path, label):
        if not (img_path.lower().endswith(IMAGE_FILE_TYPE)):
            return

        # print(img_path)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)
            self.images.append(img)
            hist = self.hog_converter.compute(img, (8, 8))
            self.hogs.append(hist.reshape(-1, 15876)[0])
            self.labels.append(label)
            self.datasize += 1
        except:
            print("corrupted image detected")

    def _limit_reached(self):
        return type(self._limit) != type(None) and self.datasize >= self._limit


if __name__ == "__main__":
    hog_converter = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    detection_dataset = DetectionDataset(hog_converter)
    print(detection_dataset.datasize)
