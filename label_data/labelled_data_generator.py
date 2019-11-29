import cv2
import sys
import os.path
import glob

# This python file is adapted from https://github.com/nagadomi/lbpcascade_animeface with some modification
# The xml file lbpcascade_animeface.xml is also got from that repository.

path = glob.glob("./tagged-anime-illustrations/moeimouto-faces/000_hatsune_miku/*.png")
path.sort()


def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(str(filename))

    split_path = str(filename).split("/")
    name = split_path[-1].split("\\")[-1]
    folder = split_path[-1].split("\\")[-2]
    print(name)
    name_path = "./labelled/" + folder + "/" + name
    folder_path = "./labelled/" + folder

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(name_path, image)

    with open(folder_path + "/" + "box.txt", "a") as f:
        f.write(name + '\n')
        for label in faces:
            if label is None:
                f.write("No label.")
            else:
                f.write(str(label) + '\n')
    f.close()

    return image

# if len(sys.argv) != 2:
#     sys.stderr.write("usage: detect.py <filename>\n")
#     sys.exit(-1)

# detect(sys.argv[1])


def label_data(input_path):
    images = []
    size = 0

    print("original number of images = ", len(input_path))

    example_path = input_path[0]
    split_path = str(example_path).split("/")
    name = split_path[-1].split("\\")[-1]
    folder = split_path[-1].split("\\")[-2]
    print(name)
    folder_path = "./labelled/" + folder
    mkdir(folder_path)

    file = open(folder_path + "/" + "box.txt", "w")
    file.close()

    for imagepath in input_path:
        size = size + 1
        label_img = detect(imagepath, cascade_file = "./lbpcascade_animeface.xml")
        if label_img is None:
            continue
        images.append(label_img)

    print("number of images = ", len(images), ", size = ", size)
    return images, size

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


if __name__=='__main__':

    images, size = label_data(path)