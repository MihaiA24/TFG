# 1. To load the dataset: image and mask paths
# 2. Building the TensorFlow Input Data Pipeline using tf.data API

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import pandas as pd


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 144

IMAGE_HEIGHT = 448
IMAGE_WIDTH = 288

def load_data(file_name):
    col_names = ['img','mask']
    df = pd.read_csv (file_name,sep=',',header=None,names=col_names)
    images = df['img'].tolist()
    masks = df['mask'].tolist()

    return images, masks

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_WIDTH,IMAGE_HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation = cv2.INTER_NEAREST)
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    return x

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)

        return x, y

    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])

    images.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    masks.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # images.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    # masks.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    return images, masks

def tf_dataset(x, y, batch_size=8, buffer_size=1000,shuffle=True):
    print(IMAGE_HEIGHT,IMAGE_WIDTH)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(preprocess)
    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset

if __name__ == "__main__":
    path = "train_dataset.csv"
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")
    images, masks = load_data(path)
    print(f"Images: {len(images)} - Masks: {len(masks)}")

    dataset = tf_dataset(images, masks)
    for x, y in dataset:
        # print(x)
        # print(y)
        # x = x[0] * 255
        # y = y[0] * 255

        x = x.numpy()
        y = y.numpy()
        print(np.unique(y))
        # cv2.imwrite("image.png", x)

        # y = np.squeeze(y, axis=-1)
        # cv2.imwrite("mask.png", y)

        break