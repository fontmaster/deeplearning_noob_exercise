# coding: utf-8
import numpy as np


class AdaGrad:
    #학습을 진행하면서 학습률을 감소시킴 (= 매개변수 전체의 학습률 값을 일괄적으로 낮춤)

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
#AdaGuard는 과거의 기울기를 제곱하여 계속 더해가는 데, 학습을 진행할 수록 갱신강도가 약해지고 무한히 학습하면 갱신량이 0이 됨
#=> RMSProp로 개선
