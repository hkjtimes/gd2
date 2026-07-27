"""Оценка агента на пачке случайных уровней заданной сложности.

    python -m gd_ai.evaluate --model checkpoints/agent.npz --episodes 50 --difficulty 0.6
"""

import argparse

import numpy as np

from .env.env import GDEnv
from .env.level_gen import LevelConfig
from .nn.model import ActorCritic


def evaluate(model, episodes=50, difficulty=0.5, seed=1000, greedy=True,
             decor_density=0.08):
    rng = np.random.default_rng(seed)
    progresses, wins = [], []
    for ep in range(episodes):
        cfg = LevelConfig(difficulty=difficulty, decor_density=decor_density)
        env = GDEnv(cfg, seed=seed + ep)
        obs = env.reset()
        done = False
        while not done:
            a, _, _ = model.act(obs[None, :], rng, greedy=greedy)
            obs, _, done, info = env.step(a[0])
        progresses.append(info["progress"])
        wins.append(float(info["won"]))
    return float(np.mean(progresses)), float(np.mean(wins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/agent.npz")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--stochastic", action="store_true",
                    help="сэмплировать действия вместо argmax")
    args = ap.parse_args()

    model = ActorCritic()
    model.load(args.model)
    progress, win_rate = evaluate(model, args.episodes, args.difficulty,
                                  args.seed, greedy=not args.stochastic)
    print(f"сложность {args.difficulty}: средний прогресс {progress:.1%}, "
          f"пройдено уровней {win_rate:.0%} из {args.episodes}")


if __name__ == "__main__":
    main()
