"""Численная проверка ручного backprop: градиенты сети сравниваются
с конечными разностями. Это главный тест корректности nn-библиотеки."""

import numpy as np
import pytest

from gd_ai.nn.layers import Dense, ReLU, Sequential, Tanh, softmax
from gd_ai.nn.model import ActorCritic


def numeric_grad(f, param, eps=1e-6):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = param[i]
        param[i] = old + eps
        f_plus = f()
        param[i] = old - eps
        f_minus = f()
        param[i] = old
        grad[i] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def test_dense_tanh_gradcheck():
    rng = np.random.default_rng(0)
    net = Sequential(Dense(5, 7, rng), Tanh(), Dense(7, 3, rng))
    x = rng.normal(size=(4, 5))
    target = rng.normal(size=(4, 3))

    def loss_fn():
        y = net.forward(x)
        return 0.5 * float(((y - target) ** 2).sum())

    y = net.forward(x)
    for _, g in net.params():
        g.fill(0.0)
    net.backward(y - target)

    for p, g in net.params():
        num = numeric_grad(loss_fn, p)
        assert np.allclose(g, num, atol=1e-5), \
            f"analytic vs numeric mismatch: {np.abs(g - num).max()}"


def test_relu_gradcheck():
    rng = np.random.default_rng(1)
    net = Sequential(Dense(4, 6, rng), ReLU(), Dense(6, 2, rng))
    x = rng.normal(size=(3, 4)) + 0.1   # избегаем точек излома ReLU
    target = rng.normal(size=(3, 2))

    def loss_fn():
        y = net.forward(x)
        return 0.5 * float(((y - target) ** 2).sum())

    y = net.forward(x)
    for _, g in net.params():
        g.fill(0.0)
    net.backward(y - target)
    for p, g in net.params():
        num = numeric_grad(loss_fn, p)
        assert np.allclose(g, num, atol=1e-5)


def test_policy_gradient_through_softmax():
    """Проверяем формулу градиента log pi(a|s) по логитам: onehot - probs."""
    rng = np.random.default_rng(2)
    B, A = 6, 2
    logits = rng.normal(size=(B, A))
    actions = rng.integers(0, A, size=B)

    def logp_sum(z):
        p = softmax(z)
        return float(np.log(p[np.arange(B), actions]).sum())

    # аналитический градиент
    probs = softmax(logits)
    grad = -probs.copy()
    grad[np.arange(B), actions] += 1.0

    eps = 1e-6
    num = np.zeros_like(logits)
    for i in range(B):
        for j in range(A):
            z1 = logits.copy(); z1[i, j] += eps
            z2 = logits.copy(); z2[i, j] -= eps
            num[i, j] = (logp_sum(z1) - logp_sum(z2)) / (2 * eps)
    assert np.allclose(grad, num, atol=1e-5)


def test_actor_critic_shapes_and_save_load(tmp_path):
    model = ActorCritic(obs_size=50, hidden=(16, 8), seed=0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(3, 50))
    logits, values = model.forward(obs)
    assert logits.shape == (3, 2) and values.shape == (3,)

    actions, logp, _ = model.act(obs, rng)
    assert set(np.unique(actions)).issubset({0, 1})
    assert np.all(logp <= 0)

    path = tmp_path / "m.npz"
    model.save(path)
    model2 = ActorCritic(obs_size=50, hidden=(16, 8), seed=99)
    model2.load(path)
    l2, v2 = model2.forward(obs)
    assert np.allclose(logits, l2) and np.allclose(values, v2)
