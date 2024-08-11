# coding: utf-8
import os
import sys

import numpy as np

from dataset.Mnist import load_mnist
from optimizer.AdaGrad import AdaGrad
from optimizer.Adam import Adam
from optimizer.Momentum import Momentum
from optimizer.SGD import SGD
from util.MultiLayerNet import MultiLayerNet
from util.Util import smooth_curve

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import matplotlib.pyplot as plt


# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000



if __name__ == '__main__':
    # 1. 실험용 설정==========
    optimizers = {}
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    # optimizers['RMSprop'] = RMSprop()

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        #5층 신경망에서, 각 층이 100개의 뉴런으로 구성, Relu 사용 (Default) -> MultiLayerNet 주석 참고..
        networks[key] = MultiLayerNet(
            input_size=784, hidden_size_list=[100, 100, 100, 100],
            output_size=10)
        train_loss[key] = []

    # 2. 훈련 시작==========
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            print("===========" + "iteration:" + str(i) + "===========")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))

    # 3. 그래프 그리기==========
    markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()