import matplotlib
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from detection_dataset import *
import pickle
from vgg16 import *

POSITIVE_DATASET_PATH_TEST="./datasets/detection/positive_test"
NEGATIVE_DATASET_PATH_TEST="./datasets/detection/negative_test"

matplotlib.use('TkAgg')


TEST_ONLY = False

class SVM:
    def __init__(self, svc, dataset=None, hog_converter=None):
        self.svc = svc

        if hog_converter:
            self.hog_converter = hog_converter

        if dataset is None:
            return

        self.dataset = dataset

        self.hog_converter = self.dataset.hog_converter

        positive_data, negative_data = np.copy(dataset.hogs[1]), np.copy(dataset.hogs[0])
        data_unscaled =  np.vstack((np.copy(positive_data), np.copy(negative_data))).astype(np.float64)
        self.scaler = StandardScaler().fit(data_unscaled)

        self.X = self.scaler.transform(data_unscaled)
        self.Y = np.asarray(self.dataset.labels[1] + self.dataset.labels[0])

    def fit(self):
        train_data, train_label = self.X, self.Y
        self.svc.fit(train_data, train_label)

    def classify(self, window):
        window_rgb = cv2.cvtColor(window, cv2.COLOR_RGB2HSV)
        window_resized = image_resize(window_rgb)
        window_hog = self.hog_converter.compute(window_resized, (8, 8))
        feature_input = [window_hog.reshape(-1, window_hog.shape[0])[0]] # window_hog.shape[0]
        res = self.svc.predict(feature_input)
        return res[0]

    def face_detection_random_test(self, test_data, test_label):
        results = self.svc.predict(test_data)

        correct = np.sum(results == test_label)

        total = test_data.shape[0]

        print("Accuracy {}".format(correct / total))




def face_detection_test(svm_face, label):
    curr_path = None

    correct = 0
    total = 0
    if label == 0:
        curr_path = NEGATIVE_DATASET_PATH_TEST
        print("Negative Test")
    else:
        curr_path = POSITIVE_DATASET_PATH_TEST
        print("Positive Test")

    file_list = listdir(curr_path)

    for path_each in file_list:
        total += 1
        img_path = join(curr_path, path_each)
        print(img_path)
        if not (img_path.lower().endswith(IMAGE_FILE_TYPE)):
            continue
        window = cv2.imread(img_path)
        result = svm_face.classify(window)
        print(result)
        if (result == label):
            correct += 1
    print(correct / total)


if __name__ == "__main__":
    hog_converter = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


    if TEST_ONLY:
        svc = pickle.load(open("./detect_svm.pkl", 'rb'))
        svm_face = SVM(svc, None, hog_converter)
        face_detection_test(svm_face, 0)
        face_detection_test(svm_face, 1)
        exit(0)




    detection_dataset = DetectionDataset(hog_converter, 10000, None) #(2000 7000), (6000, 17500), 12000
    svc = LinearSVC()

    svm_face = SVM(svc, detection_dataset)
    svm_face.fit()


    face_detection_test(svm_face, 0)
    face_detection_test(svm_face, 1)


    pkl_filename = "detect_svm.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(svm_face.svc, file)


    #run_detect()



