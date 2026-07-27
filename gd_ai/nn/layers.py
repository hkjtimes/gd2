"""Мини-библиотека нейросетей на чистом numpy.

Никаких ML-фреймворков: прямой и обратный проход написаны вручную.
Слои кэшируют вход при forward и возвращают градиенты при backward.
"""

import numpy as np


class Dense:
    """Полносвязный слой y = xW + b."""

    def __init__(self, n_in, n_out, rng, scale=None):
        # Ортогональная инициализация — стандарт для RL-политик.
        a = rng.normal(size=(n_in, n_out))
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        w = u if u.shape == (n_in, n_out) else vt
        self.W = (w * (scale if scale is not None else np.sqrt(2.0))).astype(np.float64)
        self.b = np.zeros(n_out, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW += self._x.T @ grad_out
        self.db += grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out):
        return grad_out * self._mask

    def params(self):
        return []


class Tanh:
    def __init__(self):
        self._y = None

    def forward(self, x):
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad_out):
        return grad_out * (1.0 - self._y ** 2)

    def params(self):
        return []


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def params(self):
        out = []
        for layer in self.layers:
            out.extend(layer.params())
        return out


class Adam:
    def __init__(self, params, lr=2.5e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params      # список пар (тензор, градиент)
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def zero_grad(self):
        for _, g in self.params:
            g.fill(0.0)

    def step(self, max_grad_norm=0.5):
        # Глобальный клиппинг нормы градиента.
        total = np.sqrt(sum(float((g ** 2).sum()) for _, g in self.params))
        clip = min(1.0, max_grad_norm / (total + 1e-8))
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t
        for i, (p, g) in enumerate(self.params):
            g = g * clip
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            p -= self.lr * (self.m[i] / b1t) / (np.sqrt(self.v[i] / b2t) + self.eps)


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
