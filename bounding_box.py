from detection_dataset import *
from svm import *
from shapely.geometry import Polygon

CATEGORY_NAMES_PATH = 'cat_to_name.json'

with open(CATEGORY_NAMES_PATH, 'r') as f:
    cat_to_name = json.load(f)

class BoundingBoxes:

    def __init__(self, image, image_original, scale, hog_converter, svc, window_height_min=20, window_width_min=20, window_height_max=20,
                    window_width_max=20, size_stride=1, sliding_stride=1, mode="square"):
        self.image = image
        self.image_original = image_original
        self.scale = scale
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
        self.satisfied_bounding_boxes_indices = np.copy(origin_satisfied_bounding_boxes_indices)

    def filter(self):
        origin_satisfied_bounding_boxes_indices = np.copy(self.satisfied_bounding_boxes_indices)
        current_satisfied_bounding_boxes_indices = np.copy(self.satisfied_bounding_boxes_indices)
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




    def draw_boxes(self, img_out, enlarge=2):

        for idx in self.satisfied_bounding_boxes_indices:
            x, y, w, h = self.all_bounding_boxes[idx]

            r_x, r_y = w/2, h/2
            c_x, c_y = x + r_x, y + r_y

            r_x, r_y = r_x * enlarge, r_y * enlarge

            x, y = int(c_x - r_x), int(c_y - r_y)

            w = int(enlarge * w)
            h = int(enlarge * h)



            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return img_out


    def recognition(self, model, img_out, enlarge=2):
        print("recognize")
        for idx in self.satisfied_bounding_boxes_indices:
            x, y, w, h = self.all_bounding_boxes[idx]


            r_x, r_y = w/2, h/2
            c_x, c_y = x + r_x, y + r_y

            r_x, r_y = r_x * enlarge, r_y * enlarge

            x, y = int(c_x - r_x), int(c_y - r_y)

            w = int(enlarge * w)
            h = int(enlarge * h)

            print(self.image_original.shape)

            print("({}, {}, {}, {})".format(x,y,w,h))

            window = self.image_original[y: y+h+1, x: x+w+1]
            pic = process_window(window)
            predict = predict_pic(pic, model, 10, True)
            probs = predict[0]
            classes = predict[1]
            print("classes = ", classes)

            names = []
            for i in classes:
                names.append(cat_to_name[i])

            print("==================================================================")
            print('the top possible characters are :')
            for i in range(len(names)):
                print(names[i], '( Possibility =', probs[i], ")")

            text = "{} {}".format(names[0], round(probs[0], 3))

            cv2.putText(img_out, text, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), lineType=cv2.LINE_AA)
        return img_out




def pre_process_image(img, scale):
    img_resized =  cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    img_out = np.copy(img_resized)

    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_in = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)


    return img_in, img_out



def run_detect_test():
    img = cv2.imread("./datasets/detection/detect_07.jpg")

    # 1,0.6,1,0.4, 0.3, 0.2,0.5
    # scale = 0.5

    # img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale) ))

    # img_out = np.copy(img)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    # img_in = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_in, img_out = pre_process_image(img, 0.5)

    img_out_2 = np.copy(img_out)

    hog_converter = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    svc = pickle.load(open("./detect_svm_good.pkl", 'rb'))




    bounding_boxes = BoundingBoxes(img_in, img_out, 0.5, hog_converter, svc, 80, 80, 128, 128, 16, 4) #80-128, 100-140

    bounding_boxes.sliding_window()

    bounding_boxes.get_hog_features()
    bounding_boxes.detect()



    img_out = bounding_boxes.draw_boxes(img_out, 1)

    cv2.imwrite("./datasets/detection/out_07_1.jpg", img_out)

    bounding_boxes.filter()

    img_out = bounding_boxes.draw_boxes(img_out_2, 1)

    cv2.imwrite("./datasets/detection/out_07_1_filtered.jpg", img_out)

if __name__=="__main__":
    run_detect_test()









