import glob
from torchvision import datasets, transforms
import torchvision.models as models
import cv2
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import numpy as np
from torch import nn
import sys
import os.path
import os
from torch.autograd import Variable
import torch.cuda
from collections import OrderedDict
import argparse
from torch import optim
import matplotlib.pyplot as plt
import json
import pdb
from matplotlib import cm


TEST_ONLY = True


train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

train_transforms_le = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

test_transforms_le = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

simple_transform = transforms.Compose([transforms.ToTensor()])


class LeNet(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.feature_engineering = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=5),

            nn.MaxPool2d(kernel_size=2,
                         stride=2),

            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5),

            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5,
                      out_features=120),

            nn.Linear(in_features=120,
                      out_features=84),

            nn.Linear(in_features=84,
                      out_features=10),
        )

    def forward(self, x):
        x = self.feature_engineering(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = self.avgpool(out)
        #        print(out.shape)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        #        print(out.shape)
        return out


def read_data(input_path):

    folders = []
    images = {}
    count = 0

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
                count += 1
                split_path = name_path.split("/")
                image = name
                # print("image name: ", image)
                folder = split_path[-1].split("\\")[-2]
                # print("folder name: ", folder)
                images[str(folder)].append(image)
                # print("===========================================================================================")

    return folders, images, count


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load the checkpoint or new training', default='1')
    parser.add_argument('--data_dir', help='path to the image folder', default='./datasets/recognition/train')
    parser.add_argument('--save_dir', help='path to the training checkpoint', default='./save/weights_vgg16')
    parser.add_argument('--arch', help='the architechture of the network', default='vgg')
    parser.add_argument('--lr', help='the learning rate', default=0.001)
    parser.add_argument('--hidden units', help='the hidden units', default=512)
    parser.add_argument('--epochs', help='setting the epochs', type=int, default=30)
    parser.add_argument('--device', help='CPU OR CUDA', default='cuda')
    return parser.parse_args()


def get_input_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', help='print the top N class', default=3)
    parser.add_argument('--category_names', help='the index of the labels to classes', default='cat_to_name.json')
    parser.add_argument('--device', help='CPU OR CUDA', default='cuda')
    parser.add_argument('--arch', help='the architechture of the network', default='vgg')
    parser.add_argument('--save_dir', help='path to the training checkpoint', default='./weights/weights_vgg16')
    parser.add_argument('--dirpic', help='path to the picture to test', default='./datasets/recognition/t1.png')
    return parser.parse_args()


def accuracy_test(model, dataloader):
    correct = 0
    total = 0
    model.cuda()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('the accuracy is {:.4f}'.format(correct / total))


def deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device='cuda'):
    epochs = epochs
    print_every = print_every
    steps = 0
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward and backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                print('EPOCHS : {}/{}'.format(e + 1, epochs),
                      'Loss : {:.4f}'.format(running_loss / print_every))
                accuracy_test(model, testloader)


def network_loading(model, ckp_path):
    state_dict = torch.load(ckp_path)
    model.load_state_dict(state_dict)
    print('The Network is Loaded')


def network_saving(model):
    torch.save(model.state_dict(), 'ckp')
    print('The Network is Saved')


def process_image(image):

    pic = Image.open(image)

    return process_pic(pic)

def process_window(window):
    pic = Image.fromarray(window)

    return process_pic(pic)


def process_pic(pic):
    pic = pic.resize((224, 224))
    # if pic.size[0] < pic.size[1]:
    #     ratio = float(256) / float(pic.size[0])
    # else:
    #     ratio = float(256) / float(pic.size[1])
    #
    # new_size = (int(pic.size[0] * ratio), int(pic.size[1] * ratio))
    #
    # pic.thumbnail(new_size)
    #
    # pic = pic.crop([pic.size[0] / 2 - 112, pic.size[1] / 2 - 112, pic.size[0] / 2 + 112, pic.size[1] / 2 + 112])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = np.array(pic)
    np_image = np_image / 255

    for i in range(2):
        np_image[:, :, i] -= mean[i]
        np_image[:, :, i] /= std[i]

    np_image = np_image.transpose((2, 0, 1))
    np_image = torch.from_numpy(np_image)
    np_image = np_image.float()
    np_image = np_image.type(torch.FloatTensor)
    np_image = np_image.cuda()
    return np_image


def myshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=3):

    if torch.cuda.is_available():
        model = model.cuda()
    img = process_image(image_path)
    return predict_pic(img, model, topk)


def predict_pic(img, model, topk=3, need_cuda = False):
    img = img.unsqueeze(0)
    img = img.cuda()

    if need_cuda:
        model = model.to("cuda")


    result = model(img).topk(topk)
    probs = []
    classes = []
    a = result[0]
    b = result[1].tolist()

    for i in a[0]:
        probs.append(torch.exp(i).tolist())
    for n in b[0]:
        classes.append(str(n + 1))

    return (probs, classes)


def train(net, trainloader, testloader):
    in_arg = get_input_args()

    if torch.cuda.is_available():
        net = net.cuda()

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

    # if in_arg.load == 1:
    #     network_loading(net, in_arg.save_dir)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.classifier.parameters(), lr=0.001)

    net.train()

    deep_learning(net, trainloader, testloader, in_arg.epochs, 40, criterion, optimizer, in_arg.device)

    # accuracy_test(net, testloader)

    network_saving(net)


def test(net):
    in_arg = get_input_args_predict()

    net = net.eval()

    if torch.cuda.is_available():
        net = net.cuda()

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

    network_loading(net, in_arg.save_dir)

    result = predict(in_arg.dirpic, net, in_arg.topk)
    probs = result[0]
    classes = result[1]
    print("classes = ", classes)

    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    names = []
    for i in classes:
        names.append(cat_to_name[i])

    print("==================================================================")
    print('the top possible characters are :')
    for i in range(len(names)):
        print(names[i], '( Possibility =', probs[i], ")")


if __name__ == '__main__':



    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


    if TEST_ONLY:
        vgg16 = models.vgg16(pretrained=True)
        vgg16.cuda()
        test(vgg16)
        exit(0)




    train_path = "./datasets/recognition/train"
    test_path = "./datasets/recognition/test"

    train_folders, train_images_dict, train_size = read_data(train_path)
    print("size of train = ", train_size)

    # temp = []
    # for item in train_folders:
    #     temp.append("\"" + item + "\"" + ": " + "\"" + item + "\"" + ",")
    # for item in temp:
    #     print(item)

    test_folders, test_images_dict, test_size = read_data(test_path)
    print("size of test = ", test_size)

    train_data = datasets.ImageFolder(train_path, transform=train_transforms)
    test_data = datasets.ImageFolder(test_path, transform=test_transforms)

    train_data_le = datasets.ImageFolder(train_path, transform=train_transforms_le)
    test_data_le = datasets.ImageFolder(test_path, transform=test_transforms_le)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16)

    trainloader_le = torch.utils.data.DataLoader(train_data_le, batch_size=8, shuffle=True)
    testloader_le = torch.utils.data.DataLoader(test_data_le, batch_size=8)

    my_vgg16_model = VGG16()
    my_le_model = LeNet()

    # train(my_vgg16_model, trainloader, testloader)
    test(my_vgg16_model)

    # train(my_le_model, trainloader_le, testloader_le)
    # test(my_le_model)
