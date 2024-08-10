import numpy as np


class Momentum:
    """모멘텀 SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None: #v : 물체의 속도, v는 초기화때는 아무것도 담고있지 않음.
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key] #v = 모멘텀의 속도, a = 마찰계수같은 상수값
            params[key] += self.v[key] #v = av - n(기울기), w = w + v
