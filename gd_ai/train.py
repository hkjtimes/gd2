"""Обучение агента: PPO + curriculum + доменная рандомизация.

Каждый эпизод генерируется НОВЫЙ случайный уровень, поэтому сеть не может
заучить конкретную трассу — она вынуждена научиться читать карту тайлов
и реагировать на модификаторы (режим, скорость, размер, гравитация).

Curriculum: сложность растёт автоматически, когда агент стабильно
доходит далеко. На малой сложности порталов нет; затем включаются
порталы режимов, скорости, размера и гравитации.

Запуск:
    python -m gd_ai.train --updates 300 --envs 16 --out checkpoints/agent.npz
"""

import argparse
import os
import time

import numpy as np

from .agent.ppo import PPO, PPOConfig
from .env.env import VecEnv
from .env.level_gen import LevelConfig
from .nn.model import ActorCritic


def make_config_fn(difficulty_ref, rng):
    """Фабрика конфигов уровня: случайность + текущая сложность curriculum."""

    def config_fn():
        d = difficulty_ref["value"]
        jitter = float(np.clip(d + rng.normal(0, 0.08), 0.0, 1.0))
        return LevelConfig(
            difficulty=jitter,
            allow_mode_portals=d > 0.15,
            allow_speed_portals=d > 0.25,
            allow_size_portals=d > 0.35,
            allow_gravity_portals=d > 0.45,
            decor_density=float(rng.uniform(0.0, 0.05 + 0.1 * d)),
            min_columns=120 + int(80 * d),
            max_columns=220 + int(120 * d),
        )

    return config_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/agent.npz")
    ap.add_argument("--resume", default=None, help="путь к чекпоинту для продолжения")
    ap.add_argument("--start-difficulty", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    difficulty = {"value": args.start_difficulty}

    config_fn = make_config_fn(difficulty, rng)
    venv = VecEnv(args.envs, config_fn, seed=args.seed)
    model = ActorCritic(seed=args.seed, lr=args.lr)
    if args.resume:
        model.load(args.resume)
        print(f"продолжаем с {args.resume}")

    ppo = PPO(model, venv, PPOConfig(n_steps=args.steps), seed=args.seed)

    t0 = time.time()
    for it in range(1, args.updates + 1):
        losses = ppo.train_iteration()
        s = ppo.recent_stats()

        # Curriculum: продвинулись в среднем дальше 70% уровня — усложняем.
        if s["episodes"] >= 20 and s["progress"] > 0.7:
            difficulty["value"] = min(1.0, difficulty["value"] + 0.03)

        if it % 5 == 0 or it == 1:
            sps = it * args.steps * args.envs / (time.time() - t0)
            print(f"[{it:4d}] ret={s['ret']:7.1f} progress={s['progress']:.2%} "
                  f"win={s['win_rate']:.0%} diff={difficulty['value']:.2f} "
                  f"ent={losses['entropy']:.3f} vloss={losses['v_loss']:.2f} "
                  f"({sps:.0f} steps/s, {s['episodes']} эп.)")
        if it % 25 == 0:
            model.save(args.out)

    model.save(args.out)
    print(f"готово: модель сохранена в {args.out}")


if __name__ == "__main__":
    main()
