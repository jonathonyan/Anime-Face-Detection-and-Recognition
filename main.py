from bounding_box import *
from vgg16 import *
import matplotlib.pyplot as plt


def detect_and_recognize(img, out_path, scale=0.5, detect_only = False, show = False, save=True):
    img_in, img_out = pre_process_image(img, scale)

    img_out_2 = np.copy(img_out)

    img_out, bounding_boxes = detect_main(img_in, img_out, out_path, scale, show, save)

    if not detect_only:
        img_out_2 = recognize_main(img_in, img_out_2, out_path, scale, show, save, bounding_boxes)
    return img_out_2





def detect_main(img_in, img_out, out_path, scale=0.5, show=False, save = True):
    hog_converter = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    svc = pickle.load(open("./weights/detect_svm_good.pkl", 'rb'))

    bounding_boxes = BoundingBoxes(img_in, img_out, scale, hog_converter, svc, 80, 80, 128, 128, 16, 4)


    bounding_boxes.sliding_window()

    bounding_boxes.get_hog_features()

    bounding_boxes.detect()

    bounding_boxes.filter()

    img_out = bounding_boxes.draw_boxes(img_out, 1)

    if show:
        plt.imshow(img_out)


    if save:
        cv2.imwrite(out_path, img_out)


    return img_out, bounding_boxes



def recognize_main(img_in, img_out, out_path, scale=0.5, show=False, save = True, bounding_boxes=None):

    if not bounding_boxes:
        print("Need bounding box")
        return
    vgg16 = models.vgg16(pretrained=True)

    vgg16 = pre_process_model(vgg16)

    img_out_2 = bounding_boxes.draw_boxes(img_out, 2)

    img_out_2 = bounding_boxes.recognition(vgg16, img_out_2, 2)

    if show:
        plt.imshow(img_out_2)


    if save:
        cv2.imwrite(out_path, img_out_2)


    return img_out_2





def pre_process_model(net):
    net = net.eval()

    if torch.cuda.is_available():
        net = net.to("cuda")

    for param in net.parameters():
        param.require_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(1000, 10)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    net.classifier = classifier

    network_loading(net, "./weights/weights_vgg16")
    return net








if __name__ == "__main__":
    img = cv2.imread("./datasets/detection/detect_07.jpg")


    out_path = "./datasets/detection/out_07_2.jpg"

    _ = detect_and_recognize(img, out_path, 0.5)













