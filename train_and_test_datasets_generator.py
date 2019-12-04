import cv2
import os.path


def read_data(input_path):
    folders = []
    images = {}

    for root, dirs, files in os.walk("{}".format(input_path)):

        for name in dirs:
            # print("folder name: ", name)
            folders.append(name)
            images[name] = []
            # print("folder path: ", os.path.join(root, name))
            # print("===========================================================================================")

        for name in files:
            # print("file name: ", name)
            name_path = os.path.join(root, name)
            # print("file path: ", name_path)
            if os.path.splitext(name)[1] == '.jpg' or os.path.splitext(name)[1] == '.png':
                split_path = name_path.split("/")
                image = name
                # print("image name: ", image)
                folder = split_path[-1].split("\\")[-2]
                # print("folder name: ", folder)
                images[str(folder)].append(image)
                # print("===========================================================================================")

    print("all images = ", images)
    print("all folders = ", folders)

    return folders, images


def split_data(images, path):
    train_path = "./datasets/recognition/train"
    test_path = "./datasets/recognition/test"

    for key in images:
        length = len(images[key])
        cut = int(length * 0.8)

        for i in range(length):
            image_name = images[key][i]
            image_read_path = path + "/" + key + "/" + image_name
            # print("image_read_path", image_read_path)
            # print("=======================================================================================")

            if i <= cut:
                train_folder_path = train_path + "/" + key
                mkdir(train_folder_path)
                train_write_path = train_folder_path + "/" + image_name
                # print("train_write_path", train_write_path)
                image = cv2.imread(image_read_path)
                cv2.imwrite(train_write_path, image)
                # print("=======================================================================================")

            if i > cut:
                test_folder_path = test_path + "/" + key
                mkdir(test_folder_path)
                test_write_path = test_folder_path + "/" + image_name
                # print("test_write_path", test_write_path)
                image = cv2.imread(path + "/" + key + "/" + image_name)
                cv2.imwrite(test_write_path, image)
                # print("=======================================================================================")

        print("Finished folder: ", key)
        # print("=======================================================================================")

    print("All Finished!")


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


if __name__ == '__main__':
    input_path = "./tagged-anime-illustrations/moeimouto-faces"
    folders, images = read_data(input_path)
    print("num of folders = ", len(images))
    split_data(images, input_path)
