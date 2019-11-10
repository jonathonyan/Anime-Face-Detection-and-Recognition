# Anime Face Detection and Recognition
CSC420 Fall 2019 Project

## Team Members:
Zhicheng Yan
Zhihong Wang

## Ideas:
Anime becomes more and more popular. Sometimes, people will find interesting anime characters from images without recognizing their information (e.g. the name of the character). We intend to help anime enthusiasts to find the information of characters from their anime images. For our project, we are planning to perform the anime face detection on images. It will draw a rectangular box on the face of the input anime image. After the detection, we will perform the recognition, which means our project will tell the character's information from the anime face by classification.

## Tasks:

### 1. Get Datasets
Find anime image datasets or collect anime images and create own datasets, which are large enough and contain recognizable anime faces. Datasets with labels should be preferable. We may resize them if needed.

### 2. Anime Face Detection
Detect and locate anime faces from images. When getting an anime image, the faces from the images will be highlighted by drawing rectangles. There are some options to do the face detection: finding the edges of the faces by Canny Edge Detector; using SIFT Descriptor and matching with existing anime faces from the dataset of images; detecting whether the rectangle part of the image is an anime face by training the classifier with datasets; using the shape detection tool from OpenCV.

### 3. Anime Face Recognition
When input an anime face image, we will get the information of the character from the face. We are planning to train and use the neural network (e.g. Multi-task Cascaded Convolutional Neural Network (MTCNN) or Regions with Convolutional Neural Networks (RCNN[1]), maybe with some modifications, for example, Fast RCNN) to classify the anime faces by their labels from the datasets. After training, when using the classifier on faces with different properties (e.g. sizes, colors, resolution etc.) as inputs, we may do image processing before classifying.



