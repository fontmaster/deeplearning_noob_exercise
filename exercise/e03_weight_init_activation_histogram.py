# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


if __name__ == '__main__':
    input_data = np.random.randn(1000, 100)  # 1000개의 데이터, 정규분포로 난수 생성
    node_num = 100  # 각 은닉층의 노드(뉴런) 수
    hidden_layer_size = 5  # 은닉층이 5개
    activations = {}  # 이곳에 활성화 결과를 저장

    x = input_data

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i - 1]


        w = np.random.randn(node_num, node_num) * 1
        #w = np.random.randn(node_num, node_num) * 0.01
        #w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) #Xavier
        #w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) #he , 분산의 가중치를 2/n 으로 설정

        a = np.dot(x, w)

        # 활성화 함수
        z = sigmoid(a)
        #z = ReLU(a)
        # z = tanh(a)

        activations[i] = z

    # 히스토그램 그리기
    for i, a in activations.items():
        plt.subplot(1, len(activations), i + 1)
        plt.title(str(i + 1) + "-layer")
        if i != 0: plt.yticks([], [])
        # plt.xlim(0.1, 1)
        # plt.ylim(0, 7000)
        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.show()