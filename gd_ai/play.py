"""Просмотр игры агента в ASCII-рендере прямо в терминале.

    python -m gd_ai.play --model checkpoints/agent.npz --difficulty 0.5 --fps 30
"""

import argparse
import sys
import time

import numpy as np

from .env.env import GDEnv
from .env.level_gen import LevelConfig
from .nn.model import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/agent.npz")
    ap.add_argument("--difficulty", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--random", action="store_true",
                    help="случайный агент вместо модели (для теста среды)")
    args = ap.parse_args()

    model = None
    if not args.random:
        model = ActorCritic()
        model.load(args.model)

    rng = np.random.default_rng(args.seed)
    env = GDEnv(LevelConfig(difficulty=args.difficulty), seed=args.seed)
    obs = env.reset()
    done = False
    while not done:
        if model is None:
            a = int(rng.integers(2))
        else:
            acts, _, _ = model.act(obs[None, :], rng, greedy=True)
            a = int(acts[0])
        obs, _, done, info = env.step(a)
        frame = env.render()
        sys.stdout.write("\x1b[2J\x1b[H" + frame + "\n")
        sys.stdout.flush()
        time.sleep(1.0 / args.fps)

    print("ПОБЕДА!" if info["won"] else f"смерть на {info['progress']:.0%} уровня")


if __name__ == "__main__":
    main()
