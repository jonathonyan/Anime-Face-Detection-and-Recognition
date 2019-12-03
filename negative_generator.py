import cv2
import os.path
import glob

path = glob.glob("./safebooru/bg/6000/*.jpg")
path.sort()


def detect(filename, cascade_file="./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(str(filename))

    gray = cv2.imread(str(filename), 0)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, faces


def neg_generate(filename, faces):
    image = cv2.imread(str(filename))
    neg = []
    overlap = []
    result = []

    for current_face in faces:
        (x1, y1, w1, h1) = current_face

        if y1 - h1 > 0:
            y_up = y1 - h1
            up_patch = (x1, y_up, w1, h1)
            neg.append(up_patch)

        if y1 + 2 * h1 < image.shape[0]:
            y_down = y1 + h1
            down_patch = (x1, y_down, w1, h1)
            neg.append(down_patch)

        if x1 - w1 > 0:
            x_left = x1 - w1
            left_patch = (x_left, y1, w1, h1)
            neg.append(left_patch)

        if x1 + 2 * w1 < image.shape[1]:
            x_right = x1 + w1
            right_patch = (x_right, y1 - h1, w1, h1)
            neg.append(right_patch)

        if len(neg) == 0:
            continue

    for patch in neg:
        for rest_face in faces:
            (x2, y2, w2, h2) = rest_face
            x = patch[0]
            y = patch[1]
            w = patch[2]
            h = patch[3]
            if (x + w) > x2 and (y + h) > y2 and x < (x2 + w2) and y < (y2 + h2):
                overlap.append(patch)
                break

    for bad_patch in overlap:
        neg.remove(bad_patch)

    for patch in neg:
        x = patch[0]
        y = patch[1]
        w = patch[2]
        h = patch[3]
        temp = image[y:y + h, x:x + w]
        result.append(temp)

    return image, neg, result


def negative_generator(input_path):
    result = []
    size = 0

    print("original number of images = ", len(input_path))

    for imagepath in input_path:
        size = size + 1
        label_img, faces = detect(imagepath, cascade_file="./lbpcascade_animeface.xml")
        if label_img is None:
            continue
        neg_image, neg, neg_patches = neg_generate(imagepath, faces)
        if len(neg_patches) == 0:
            continue
        for patch in neg_patches:
            result.append(patch)

    print("num of total neg patches = ", len(result))

    i = 11389
    for patch in result:
        cv2.imwrite("negative/6000/{}.jpg".format(i), patch)
        i += 1

    return result


def simple_neg_generator(input_path):
    size = 0
    i = 17264

    print("original number of images = ", len(input_path))

    for imagepath in input_path:
        size = size + 1
        image = cv2.imread(str(imagepath))
        if image is None:
            continue
        row = image.shape[0]
        col = image.shape[1]
        length = int(row / 10)
        top_left = image[0:length, 0:length]
        top_right = image[0:length, col - length:col]
        bottom_left = image[row - length:row, 0:length]
        bottom_right = image[row - length:row, col - length:col]
        cv2.imwrite("negative/corner/{}.jpg".format(i), top_left)
        cv2.imwrite("negative/corner/{}.jpg".format(i + 1), top_right)
        cv2.imwrite("negative/corner/{}.jpg".format(i + 2), bottom_left)
        cv2.imwrite("negative/corner/{}.jpg".format(i + 3), bottom_right)
        i = i + 4


if __name__ == '__main__':
    # neg_images = negative_generator(path)
    simple_neg_generator(path)
