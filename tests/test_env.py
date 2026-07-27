"""Тесты среды: генерация уровней, физика режимов, порталы, декорации."""

import numpy as np
import pytest

from gd_ai.env import constants as C
from gd_ai.env.env import GDEnv, VecEnv
from gd_ai.env.level_gen import LevelConfig, LevelGenerator


def test_level_generation_all_difficulties():
    for d in (0.0, 0.3, 0.6, 1.0):
        for seed in range(5):
            gen = LevelGenerator(LevelConfig(difficulty=d),
                                 np.random.default_rng(seed))
            grid, meta = gen.generate()
            assert grid.shape[0] == C.LEVEL_HEIGHT
            assert grid.shape[1] == meta["width"]
            assert (grid[:, -1] == C.FINISH).all()
            # стартовая зона проходима: нет шипов и блоков в первых колонках
            start = grid[:, :4]
            assert not np.isin(start, (C.SPIKE, C.BLOCK)).any()


def test_observation_shape_and_range():
    env = GDEnv(LevelConfig(difficulty=0.5), seed=0)
    obs = env.reset()
    assert obs.shape == (C.OBS_SIZE,)
    assert np.isfinite(obs).all()
    obs, r, d, info = env.step(1)
    assert obs.shape == (C.OBS_SIZE,)
    assert np.isfinite(r)


def test_decor_never_kills():
    """Уровень целиком из декораций должен проходиться при любых действиях."""
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[:8, ::2] = C.DECOR         # плотный "лес" декора
    env.grid[:, -1] = C.FINISH
    rng = np.random.default_rng(0)
    done = False
    while not done:
        _, _, done, info = env.step(int(rng.integers(2)))
    assert info["won"], "декорация убила игрока или помешала пройти"


def test_spike_kills_cube():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[0, 6] = C.SPIKE
    env.grid[:, -1] = C.FINISH
    done = False
    while not done:
        _, _, done, info = env.step(0)   # никогда не прыгаем
    assert not info["won"] and env.dead


def test_cube_jumps_over_spike():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[0, 8] = C.SPIKE
    env.grid[:, -1] = C.FINISH
    done = False
    while not done:
        # прыгаем, когда шип близко
        dist = 8 - env.x
        a = 1 if 0 < dist < 1.8 and env._check_support(env._size()) else 0
        _, _, done, info = env.step(a)
    assert info["won"], f"куб не перепрыгнул шип, x={env.x:.1f}"


def test_mode_portal_changes_mode():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[0:5, 5] = C.PORTAL_SHIP
    env.grid[:, -1] = C.FINISH
    for _ in range(60):
        _, _, done, _ = env.step(0)
        if done:
            break
    assert env.mode == C.MODE_SHIP


def test_speed_and_size_portals():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[0:5, 5] = C.SPEED_3X
    env.grid[0:5, 12] = C.SIZE_MINI
    env.grid[:, -1] = C.FINISH
    for _ in range(200):
        _, _, done, _ = env.step(0)
        if done or (env.speed_portal == C.SPEED_3X and env.mini):
            break
    assert env.speed_portal == C.SPEED_3X
    assert env.mini
    assert env._size() == C.PLAYER_SIZE_MINI


def test_gravity_portal_flips():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[0:5, 5] = C.GRAV_FLIP
    env.grid[:, -1] = C.FINISH
    for _ in range(100):
        env.step(0)
        if env.gdir == -1:
            break
    assert env.gdir == -1
    # с перевёрнутой гравитацией игрок "падает" к потолку
    for _ in range(100):
        if env.dead or env.won:
            break
        env.step(0)
        if env.y + env._size() >= C.LEVEL_HEIGHT - 0.01:
            break
    assert env.y + env._size() >= C.LEVEL_HEIGHT - 0.5


def test_ship_can_fly():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[:, -1] = C.FINISH
    env.mode = C.MODE_SHIP
    y0 = env.y
    for _ in range(40):
        env.step(1)   # держим — корабль набирает высоту
    assert env.y > y0 + 1.0


def test_wave_dies_on_block():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[0:12, 8] = C.BLOCK    # стена
    env.grid[:, -1] = C.FINISH
    env.mode = C.MODE_WAVE
    done = False
    for _ in range(300):
        _, _, done, _ = env.step(0)
        if done:
            break
    assert env.dead


def test_spider_teleports_to_ceiling():
    env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=0)
    env.grid[:] = C.EMPTY
    env.grid[:, -1] = C.FINISH
    env.mode = C.MODE_SPIDER
    env.step(0)
    y_before = env.y
    env.step(1)      # телепорт
    assert env.gdir == -1
    assert env.y > y_before + 5.0   # улетел к потолку


def test_all_modes_random_agent_survives_empty_level():
    """В пустом уровне ни один режим не должен умирать (пол/потолок безопасны)."""
    for mode in range(C.N_MODES):
        env = GDEnv(LevelConfig(difficulty=0.0, decor_density=0.0), seed=mode)
        env.grid[:] = C.EMPTY
        env.grid[:, -1] = C.FINISH
        env.mode = mode
        rng = np.random.default_rng(mode)
        done = False
        while not done:
            _, _, done, info = env.step(int(rng.integers(2)))
        assert info["won"], f"режим {C.MODE_NAMES[mode]} умер в пустом уровне"


def test_vec_env_autoreset():
    venv = VecEnv(4, lambda: LevelConfig(difficulty=0.3), seed=0)
    obs = venv.reset()
    assert obs.shape == (4, C.OBS_SIZE)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, r, d, infos = venv.step(rng.integers(0, 2, size=4))
        assert obs.shape == (4, C.OBS_SIZE)
        assert np.isfinite(obs).all()


def test_random_levels_are_survivable_at_start():
    """Первые 30 тиков любой сгенерированный уровень не убивает бездействием."""
    for seed in range(20):
        env = GDEnv(LevelConfig(difficulty=0.8), seed=seed)
        for _ in range(30):
            _, _, done, _ = env.step(0)
            if done:
                break
        assert not env.dead, f"уровень seed={seed} убивает в стартовой зоне"
