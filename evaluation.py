# -*- coding: utf-8 -*-


import numpy as np


def evaluation(pred, truth):

    nums = len(pred)

    correct = 0
    for i in range(nums):

        pred_id = np.argmax(pred[i])

        if pred_id == truth[i]:
            correct += 1

    acc = float(correct)/float(nums)

    print("TEST ACC: %s" % acc)

    return acc