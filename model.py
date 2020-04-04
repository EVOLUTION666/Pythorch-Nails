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


def show_box(image_name, data):
    img = os.path.join('images/train/', image_name)
    rows_by_img = data.loc[data['filename'] == image_name]
    image = cv2.imread(img)

    for index, row in rows_by_img.iterrows():
        start_point = (int(row['xmin']), int(row['ymin']))
        end_point = (int(row['xmax']), int(row['ymax']))

        print(start_point)
        print(end_point)
        print(image_name)

        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)


show_box('93.jpg', test)