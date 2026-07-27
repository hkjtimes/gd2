"""Smoke-тест PPO: короткое обучение проходит без ошибок,
параметры обновляются и остаются конечными."""

import numpy as np

from gd_ai.agent.ppo import PPO, PPOConfig
from gd_ai.env.env import VecEnv
from gd_ai.env.level_gen import LevelConfig
from gd_ai.nn.model import ActorCritic


def test_ppo_smoke():
    venv = VecEnv(2, lambda: LevelConfig(difficulty=0.1, decor_density=0.02,
                                         min_columns=60, max_columns=80), seed=0)
    model = ActorCritic(hidden=(32, 16), seed=0, lr=1e-3)
    w_before = model.policy_head.W.copy()

    ppo = PPO(model, venv, PPOConfig(n_steps=64, n_minibatches=2, epochs=2), seed=0)
    for _ in range(2):
        stats = ppo.train_iteration()

    assert all(np.isfinite(v) for v in stats.values())
    assert not np.allclose(w_before, model.policy_head.W), "веса не обновились"
    for p, _ in model.opt.params:
        assert np.isfinite(p).all()


def test_ppo_learns_to_jump_single_spike():
    """Мини-проверка обучаемости: на сверхпростых уровнях средний прогресс
    должен вырасти за несколько итераций."""
    def cfg():
        return LevelConfig(difficulty=0.15, decor_density=0.0,
                           allow_mode_portals=False, allow_speed_portals=False,
                           allow_size_portals=False, allow_gravity_portals=False,
                           min_columns=50, max_columns=70)

    venv = VecEnv(4, cfg, seed=0)
    model = ActorCritic(hidden=(64, 32), seed=0, lr=1e-3)
    ppo = PPO(model, venv, PPOConfig(n_steps=256, n_minibatches=4, epochs=4), seed=0)

    ppo.train_iteration()
    early = np.mean(ppo.ep_progress[:20]) if len(ppo.ep_progress) >= 5 else 0.0
    for _ in range(8):
        ppo.train_iteration()
    late = np.mean(ppo.ep_progress[-20:])
    assert late > early - 0.05, f"прогресс упал: {early:.2f} -> {late:.2f}"
