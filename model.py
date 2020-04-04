import torchvision
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import os
import cv2

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)

test = pd.read_csv('images/test_labels.csv')
train = pd.read_csv('images/train_labels.csv')
test_count = len(test) - 1;
train_count = len(train) - 1;

img_name = train.iloc[train_count, 0]
box_coordinates = train.iloc[train_count, 4:]

box_coordinates = np.asarray(box_coordinates)
box_coordinates = box_coordinates.astype('float').reshape(-1, 2)


def show_box(box_coordinates, image_name):
    img = os.path.join('images/train/', image_name)

    start_points = (int(box_coordinates[0][0]), int(box_coordinates[0][1]))
    end_points = (int(box_coordinates[1][0]), int(box_coordinates[1][1]))

    image = cv2.imread(img)
    cv2.rectangle(image, start_points, end_points, (0, 0, 255), 2)
    # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)

show_box(box_coordinates, img_name)