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


def show_box(image_name, folder):
    img = os.path.join('images/' + folder + '/', image_name)

    if folder == 'test':
        data = test
    else:
        data = train

    if os.path.isfile(img):
        rows_by_img = data.loc[data['filename'] == image_name]
        image = cv2.imread(img)

        for index, row in rows_by_img.iterrows():
            start_point = (int(row['xmin']), int(row['ymin']))
            end_point = (int(row['xmax']), int(row['ymax']))
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)

        cv2.imshow("Output", image)
        cv2.waitKey(0)
    else:
        print("Image not exist")


class NailDataset(object):
    def __init__(self, img_name, folder, transforms):
        self.img_name = img_name
        self.folder = folder
        self.transforms = transforms

    def __getitem__(self, idx):
        123



show_box('637.jpg', 'test')