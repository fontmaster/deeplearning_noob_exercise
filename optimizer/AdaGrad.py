# coding: utf-8
import numpy as np


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # 행렬곱을 통해 학습률을 조정
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

