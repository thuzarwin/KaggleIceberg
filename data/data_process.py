# -*- coding: utf-8 -*-

import json
import pickle
import numpy as np
import random


def preprocess_train_data():
    """
    Convert JSON train data to pkl
    :param filename:
    :return:
    """

    f = open('train.json', 'r')

    raw_data = json.load(f)

    f.close()

    def get_record(x):

        band_image_1 = np.array(x['band_1'])
        band_image_2 = np.array(x['band_2'])
        band_image_1 = band_image_1.reshape((75, 75))
        band_image_2 = band_image_2.reshape((75, 75))
        image = np.stack([band_image_1, band_image_2])
        label = x['is_iceberg']
        return image, label

    train_images = []
    train_labels = []

    for i in range(len(raw_data)):

        image, label = get_record(raw_data[i])
        train_labels.append(label)
        train_images.append(image)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    with open('train_data.pkl', 'wb') as ff:

        pickle.dump(train_images, ff)

    with open('train_label.pkl', 'wb') as ff:

        pickle.dump(train_labels, ff)

    print("Finish Preprocess Train Data")


def load_train_data(path):

    with open(path+'/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(path+'/train_label.pkl', 'rb') as f:
        train_label = pickle.load(f)

    train_data = zip(train_data, train_label)
    num_samples = len(train_data)
    ratio = 0.9
    num_train = int(num_samples*ratio)
    random.shuffle(train_data)
    train_samples = train_data[:num_train]
    test_samples = train_data[num_train:]

    return train_samples, test_samples


def load_test_data(path):

    """
    Load Test JSON data
    :return:
    """

    f = open(path+'/test.json', 'r')

    raw_data = json.load(f)

    f.close()

    def get_image(x):

        image_id = x['id']
        band_image_1 = np.array(x['band_1'])
        band_image_2 = np.array(x['band_2'])
        band_image_1 = band_image_1.reshape((75, 75))
        band_image_2 = band_image_2.reshape((75, 75))
        image = np.stack([band_image_1, band_image_2])
        return image_id, image

    for i in range(len(raw_data)):

        image_id, image = get_image(raw_data[i])

        yield {
            'image_id': image_id,
            'image': image
        }


# if __name__ == '__main__':
#     preprocess_train_data()
#
#     train_data, test_data = load_train_data()
#
#     print(train_data[10])







