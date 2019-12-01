from detection_dataset import *
from svm import *
from shapely.geometry import Polygon


class BoundingBoxes:

    def __init__(self, image, hog_converter, svc, window_height_min=20, window_width_min=20, window_height_max=20,
                    window_width_max=20, size_stride=1, sliding_stride=1, mode="square"):
        self.image = image
        self.hog_converter = hog_converter
        self.svc = svc
        self.all_bounding_boxes = []
        self.all_hog_features = []
        self.results = []
        self.satisfied_bounding_boxes_indices = []
        self.window_height_min = window_height_min
        self.window_height_max = window_height_max
        self.window_width_min = window_width_min
        self.window_width_max = window_width_max
        self.size_stride = size_stride
        self.sliding_stride = sliding_stride
        self.mode = mode

    def sliding_window(self):
        if self.mode == "square":
            self._sliding_window_square()
        else:
            self._sliding_window_rectangle()



    def _sliding_window_rectangle(self):
        if self.image.shape[0] <= self.window_height_min or self.image.shape[1] <= self.window_width_min:
            print("The image is too small")
            return

        w_min = self.window_width_min
        h_min = self.window_height_min
        w_max = min(self.window_width_max, self.image.shape[1])
        h_max = min(self.window_height_max, self.image.shape[0])

        for w in range(w_min, w_max, self.size_stride):
            for h in range(h_min, h_max, self.size_stride):
                x_range = np.arange(0, self.image.shape[1] - w, self.sliding_stride)
                y_range = np.arange(0, self.image.shape[0] - h, self.sliding_stride)
                x_y = np.transpose([np.tile(x_range, len(y_range)), np.repeat(y_range, len(x_range))])
                bounding_boxes_curr = np.hstack((x_y, np.ones((x_y.shape[0],1)) * w,  np.ones((x_y.shape[0],1))*h))
                self.all_bounding_boxes.extend(bounding_boxes_curr)


    def _sliding_window_square(self):
        side_min = self.window_height_min
        side_max = np.min([self.window_height_max, self.image.shape[0], self.image.shape[1]])


        for s in range(side_min, side_max, self.size_stride):
            x_range = np.arange(0, self.image.shape[1] - s, self.sliding_stride)
            y_range = np.arange(0, self.image.shape[0] - s, self.sliding_stride)
            x_y = np.transpose([np.tile(x_range, len(y_range)), np.repeat(y_range, len(x_range))])
            bounding_boxes_curr = np.hstack((x_y, np.ones((x_y.shape[0],1)) * s,  np.ones((x_y.shape[0],1))*s))
            self.all_bounding_boxes.extend(bounding_boxes_curr.astype(np.int))



    def get_hog_features(self):
        for box in self.all_bounding_boxes:
            x, y, w, h = box
            window = self.image[y: y+h+1, x: x+w+1]
            window_resized = image_resize(window)
            hist = self.hog_converter.compute(window_resized, (8, 8))
            window_hog = hist.reshape(-1, hist.shape[0])[0]
            self.all_hog_features.append(window_hog)

    def detect(self):
        for i in range(0, len(self.all_bounding_boxes), 10000):
            hog_features_batch = self.all_hog_features[i: i+10000]
            results_batch = self.svc.predict(hog_features_batch)
            self.results.extend(results_batch)
        self.results = np.array(self.results)
        origin_satisfied_bounding_boxes_indices = np.where(self.results == 1)[0]
        current_satisfied_bounding_boxes_indices = np.copy(origin_satisfied_bounding_boxes_indices)

        for idx in origin_satisfied_bounding_boxes_indices:
            overlap = False
            x1, y1, w1, h1 = self.all_bounding_boxes[idx]
            bb1 = Polygon([[x1, y1], [x1 + w1, y1], [x1+w1, y1+h1], [x1, y1+h1]])
            for idx2 in current_satisfied_bounding_boxes_indices:
                if idx2 == idx:
                    continue
                x2, y2, w2, h2 = self.all_bounding_boxes[idx2]
                bb2 = Polygon([[x2, y2], [x2 + w2, y2], [x2+w2, y2+h2], [x2, y2+h2]])
                iou = bb1.intersection(bb2).area / bb1.union(bb2).area
                if iou > 0.3:
                    overlap = True

            if overlap:
                self.results[idx] = 0
            current_satisfied_bounding_boxes_indices = np.where(self.results == 1)[0]

        self.satisfied_bounding_boxes_indices = np.copy(current_satisfied_bounding_boxes_indices)








    def _get_iou(self, bb1, bb2):
        poly_1 = Polygon(bb1)
        poly_2 = Polygon(bb2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou




    def draw_boxes(self, img_out):

        for idx in self.satisfied_bounding_boxes_indices:
            x, y, w, h = self.all_bounding_boxes[idx]
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return img_out



def run_detect():
    img = cv2.imread("./datasets/detection/detect_05.jpg")

    # 1,0.6,1,0.4, 0.3, 0.2,0.5
    scale = 0.2

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale) ))

    img_out = np.copy(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hog_converter = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    svc = pickle.load(open("./detect_svm_good.pkl", 'rb'))




    bounding_boxes = BoundingBoxes(img_hsv, hog_converter, svc, 80, 80, 128, 128, 16, 4) #80-128, 100-140

    bounding_boxes.sliding_window()

    bounding_boxes.get_hog_features()
    bounding_boxes.detect()



    img_out = bounding_boxes.draw_boxes(img_out)

    cv2.imwrite("./datasets/detection/out_05.jpg", img_out)

if __name__=="__main__":
    run_detect()









