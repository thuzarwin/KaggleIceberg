# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.optim as optimizer
import argparse
import random
from model import cnn
from data import data_process as dp
import evaluation
import output


def train(gpu, batch_size, epoches):

    train_data, test_data = dp.load_train_data('data')
    model = cnn.CNNet()
    criterion = torch.nn.CrossEntropyLoss()

    if gpu:
        model = model.cuda()
        criterion = criterion.cuda()
    num_samples = len(train_data)

    learning_rate = 0.00005

    train_optimizer = optimizer.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.001)

    for i in range(epoches):
        print("Epoch: %s:" % (i + 1))
        random.shuffle(train_data)
        sum_loss = 0
        for j in range(num_samples/batch_size):

            images = []
            labels = []

            for sample_index in range(batch_size):
                images.append(train_data[sample_index+j][0])
                labels.append(train_data[sample_index+j][1])

            images = Variable(torch.FloatTensor(images))
            labels = Variable(torch.LongTensor(labels))

            if gpu:
                images = images.cuda()
                labels = labels.cuda()

            prob = model(images)
            loss = criterion(prob, labels)
            sum_loss += loss.cpu().data.numpy()[0]
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            if j % 5 == 0:
                print("Batch Index: %s Loss: %s Avg Loss: %s" % (j, loss.cpu().data.numpy()[0], sum_loss/(j+1)))

        print("Epoch: %s Loss: %s" % (i, sum_loss/(num_samples/batch_size)))

        torch.save(model.state_dict(), 'model_params/epoch_%s_params.model' % i)
        labels, result = test(gpu, test_data, 'model_params/epoch_%s_params.model % i')
        acc = evaluation.evaluation(result, labels)
        print('TEST ACC: %s',acc)


def test(gpu, dataset, model_path):

    model = cnn.CNNet()
    if gpu:
        model = model.cuda()
    model.load_state_dict(model_path)

    num_samples = len(dataset)

    labels = []
    result = []
    for i in range(num_samples):
        image = dataset[i][0]
        labels.append(dataset[i][1])

        image = Variable(torch.FloatTensor(image))
        if gpu:
            image = image.cuda()
        image = image.unsqueeze(0)
        prob = model(image)
        prob = prob.squeeze(0)
        prob = prob.cpu().data.numpy()
        result.append(prob)

    return labels, result


def predict(x, model, gpu):

    image = Variable(torch.FloatTensor(x))
    if gpu:
        image = image.cuda()
    image = image.unsqueeze(0)
    prob = model(image)
    prob = prob.squeeze(0)
    prob = prob.cpu().data.numpy()

    return prob[1]


def get_predict(gpu, idx):

    # model = torch.load('model_params/epoch_8_params.model')
    model = cnn.CNNet()
    model.load_state_dict('model_params/epoch_%s_params.model' % idx)

    result = []
    for sample in dp.load_test_data('data'):
        prob = predict(sample['image'], model, gpu)
        result.append((sample['image_id'], prob))

    output.output('output/submit.csv', result)


# if __name__ == '__main__':
#
#     train(16, 20)

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--train', type=int, default=1, help='Traing')
    args.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    args.add_argument('--epoches', type=int, default=20, help='training epoches')
    args.add_argument('--gpu', type=int, default=0, help='Use GPU')
    args.add_argument('--idx', type=int, default=10, help='Params id')

    params = args.parse_args()

    if params.train:

        train(params.gpu, params.batch_size, params.epoches)
    else:
        get_predict(params.gpu, params.idx)

