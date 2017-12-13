# -*- coding: utf-8 -*-

from model import encoder_gbm
import torch.optim as optimizer
import torch
from torch.autograd import Variable
from data import data_process as dp
import random


def train_encoder(gpu, batch_size, epoches):

    train_data, test_data = dp.load_train_data('data')

    criterion = torch.nn.MSELoss()
    encoder = encoder_gbm.Encoder()
    decoder = encoder_gbm.Decoder(512)
    if gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()

    num_samples = len(train_data)

    encoder_optimizer = optimizer.Adam(lr=0.0001, params=encoder.parameters())
    decoder_optimizer = optimizer.Adam(lr=0.0001, params=decoder.parameters())

    for i in range(epoches):
        print("Epoch: %s:" % (i + 1))
        random.shuffle(train_data)
        sum_loss = 0
        for j in range(num_samples/batch_size):

            images = []

            for sample_index in range(batch_size):
                images.append(train_data[sample_index+j][0])

            images = Variable(torch.FloatTensor(images))

            if gpu:
                images = images.cuda()

            encoded = encoder(images)
            decoded = decoder(encoded)
            loss = criterion(decoded, images)
            sum_loss += loss.cpu().data.numpy()[0]
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()

            if j % 5 == 0:
                print("Batch Index: %s Loss: %s Avg Loss: %s" % (j, loss.cpu().data.numpy()[0], sum_loss/(j+1)))

        print("Epoch: %s Loss: %s" % (i, sum_loss/(num_samples/batch_size)))
        torch.save(encoder.state_dict(), 'model_params/encoder_%s.model' % i)


train_encoder(False ,1, 10)