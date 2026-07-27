"""Модель актор-критик для агента Geometry Dash.

Общий ствол-MLP обрабатывает наблюдение (локальная карта тайлов + состояние
игрока с режимом/скоростью/размером/гравитацией), две головы:
  * policy — логиты действий (отпустить / зажать кнопку),
  * value  — оценка ценности состояния для PPO.
"""

import numpy as np

from ..env import constants as C
from .layers import Adam, Dense, ReLU, Sequential, Tanh, softmax


class ActorCritic:
    def __init__(self, obs_size=C.OBS_SIZE, n_actions=C.N_ACTIONS,
                 hidden=(256, 128), seed=0, lr=2.5e-4):
        rng = np.random.default_rng(seed)
        trunk = []
        n_in = obs_size
        for h in hidden:
            trunk += [Dense(n_in, h, rng), Tanh()]
            n_in = h
        self.trunk = Sequential(*trunk)
        self.policy_head = Dense(n_in, n_actions, rng, scale=0.01)
        self.value_head = Dense(n_in, 1, rng, scale=1.0)
        self.opt = Adam(self.trunk.params() + self.policy_head.params()
                        + self.value_head.params(), lr=lr)

    # ---------------------------------------------------------------- forward

    def forward(self, obs):
        """obs: [B, obs_size] → (logits [B, A], values [B])."""
        z = self.trunk.forward(obs)
        logits = self.policy_head.forward(z)
        values = self.value_head.forward(z)[:, 0]
        return logits, values

    def act(self, obs, rng, greedy=False):
        """Сэмплирует действия. obs: [B, obs_size]."""
        logits, values = self.forward(obs)
        probs = softmax(logits)
        if greedy:
            actions = probs.argmax(axis=-1)
        else:
            u = rng.random((obs.shape[0], 1))
            actions = (u > probs[:, :1]).astype(np.int64)[:, 0]  # 2 действия
        logp = np.log(probs[np.arange(len(actions)), actions] + 1e-10)
        return actions, logp, values

    # --------------------------------------------------------------- backward

    def backward_heads(self, grad_logits, grad_values):
        """Обратный проход через обе головы и ствол."""
        g_trunk = self.policy_head.backward(grad_logits)
        g_trunk += self.value_head.backward(grad_values[:, None])
        self.trunk.backward(g_trunk)

    # ------------------------------------------------------------- сохранение

    def save(self, path):
        arrays = {}
        for i, (p, _) in enumerate(self.opt.params):
            arrays[f"p{i}"] = p
        np.savez_compressed(path, **arrays)

    def load(self, path):
        data = np.load(path)
        for i, (p, _) in enumerate(self.opt.params):
            np.copyto(p, data[f"p{i}"])
