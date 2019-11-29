import matplotlib
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from detection_dataset import *


matplotlib.use('TkAgg')
class SVM:
    def __init__(self, dataset, svc):
        self.dataset = dataset
        self.svc = svc

        positive_data, negative_data = np.copy(dataset.hogs[1]), np.copy(dataset.hogs[0])
        data_unscaled =  np.vstack((np.copy(positive_data), np.copy(negative_data))).astype(np.float64)
        self.scaler = StandardScaler().fit(data_unscaled)

        self.X = self.scaler.transform(data_unscaled)
        self.Y = np.asarray(self.dataset.labels[1] + self.dataset.labels[0])

    def fit(self):
        train_data, test_data, train_label, test_label = train_test_split(self.X, self.Y, test_size=0.1)
        self.svc.fit(train_data, train_label)



if __name__ == "__main__":
    hog_converter = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    detection_dataset = DetectionDataset(hog_converter)
    svc = LinearSVC()
    svm_face = SVM(detection_dataset, svc)
    svm_face.fit()
