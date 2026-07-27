"""Среда Geometry Dash в стиле gym: reset()/step(action).

Игрок движется вправо с постоянной скоростью (зависит от модификатора
скорости). Управление — одна кнопка (0 = отпущена, 1 = зажата), как в GD.
Поддерживаются все 7 режимов (куб, корабль-ракета, шар, НЛО, волна, робот,
паук), порталы скорости 0.5x-4x, мини-размер и переворот гравитации.

Наблюдение = локальное окно тайлов вокруг игрока (5 каналов) + вектор
состояния (режим, скорость, размер, гравитация, вертикальная скорость...).
Декорации попадают в отдельный канал наблюдения: сеть видит их, но должна
научиться игнорировать.
"""

import numpy as np

from . import constants as C
from .level_gen import LevelConfig, LevelGenerator


class GDEnv:
    def __init__(self, config: LevelConfig | None = None, max_ticks=6000, seed=None):
        self.cfg = config or LevelConfig()
        self.max_ticks = max_ticks
        self.rng = np.random.default_rng(seed)
        self.grid = None
        self.reset()

    # ------------------------------------------------------------------ API

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        gen = LevelGenerator(self.cfg, self.rng)
        self.grid, self.meta = gen.generate()
        self.width = self.grid.shape[1]

        self.x = 1.5
        self.y = 0.0
        self.vy = 0.0
        self.gdir = 1                 # 1 — обычная гравитация, -1 — перевёрнутая
        self.mode = C.MODE_CUBE
        self.speed_portal = C.SPEED_1X
        self.mini = False
        self.prev_action = 0
        self.hold_ticks = 0
        self.ticks = 0
        self.dead = False
        self.won = False
        self.best_x = self.x
        return self._observe()

    def step(self, action):
        """action ∈ {0, 1}. Возвращает (obs, reward, done, info)."""
        assert not (self.dead or self.won), "call reset() after episode end"
        action = int(action)
        prev_x, prev_y = self.x, self.y
        size = self._size()
        hspeed = C.SPEED_VALUES[self.speed_portal]

        self.x += hspeed
        grounded = self._check_support(size)
        self._apply_mode_physics(action, grounded, hspeed, size)
        self.y += self.vy

        # Пол и потолок мира: не убивают, по ним можно скользить.
        if self.y < 0.0:
            self.y = 0.0
            if self.vy < 0:
                self.vy = 0.0
        if self.y + size > C.LEVEL_HEIGHT:
            self.y = C.LEVEL_HEIGHT - size
            if self.vy > 0:
                self.vy = 0.0

        self._resolve_tiles(prev_y, size)

        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.dead = True

        progress = max(0.0, self.x - self.best_x)
        self.best_x = max(self.best_x, self.x)
        reward = progress
        if self.dead:
            reward -= 3.0
        if self.won:
            reward += 10.0

        done = self.dead or self.won
        info = {
            "x": self.x, "progress": self.x / self.width,
            "won": self.won, "mode": self.mode,
        }
        self.prev_action = action
        return self._observe(), reward, done, info

    # ------------------------------------------------------------- физика

    def _size(self):
        return C.PLAYER_SIZE_MINI if self.mini else C.PLAYER_SIZE

    def _check_support(self, size):
        """Есть ли опора в направлении гравитации (для прыжка/флипа)."""
        eps = 0.06
        if self.gdir == 1:
            if self.y <= eps:
                return True
            probe_y = self.y - eps
        else:
            if self.y + size >= C.LEVEL_HEIGHT - eps:
                return True
            probe_y = self.y + size + eps - 1e-9
        row = int(np.floor(probe_y))
        if not (0 <= row < C.LEVEL_HEIGHT):
            return False
        for col in self._cols(size):
            if self.grid[row, col] == C.BLOCK:
                return True
        return False

    def _cols(self, size, shrink=0.02):
        c0 = int(np.floor(self.x - size / 2 + shrink))
        c1 = int(np.floor(self.x + size / 2 - shrink))
        return [c for c in (c0, c1) if 0 <= c < self.width] if c1 != c0 else \
            ([c0] if 0 <= c0 < self.width else [])

    def _apply_mode_physics(self, action, grounded, hspeed, size):
        press_edge = action == 1 and self.prev_action == 0
        g = self.gdir
        m = self.mode

        if m == C.MODE_CUBE:
            self.vy -= g * C.GRAVITY_CUBE
            if action == 1 and grounded:
                jump = C.JUMP_CUBE * (C.MINI_JUMP_SCALE if self.mini else 1.0)
                self.vy = g * jump
        elif m == C.MODE_ROBOT:
            self.vy -= g * C.GRAVITY_ROBOT
            if action == 1 and grounded and self.hold_ticks == 0:
                self.vy = g * C.JUMP_ROBOT
                self.hold_ticks = 1
            elif action == 1 and 0 < self.hold_ticks <= C.ROBOT_MAX_HOLD:
                self.vy += g * C.ROBOT_HOLD_BOOST
                self.hold_ticks += 1
            elif action == 0 and grounded:
                self.hold_ticks = 0
        elif m == C.MODE_SHIP:
            self.vy += g * (C.SHIP_THRUST if action == 1 else 0.0)
            self.vy -= g * C.GRAVITY_SHIP
            self.vy = float(np.clip(self.vy, -C.SHIP_MAX_VY, C.SHIP_MAX_VY))
        elif m == C.MODE_UFO:
            self.vy -= g * C.GRAVITY_UFO
            if press_edge:
                self.vy = g * C.UFO_IMPULSE
        elif m == C.MODE_WAVE:
            steep = 1.4 if self.mini else 1.0
            self.vy = g * hspeed * steep * (1.0 if action == 1 else -1.0)
        elif m == C.MODE_BALL:
            self.vy -= g * C.GRAVITY_BALL
            if press_edge and grounded:
                self.gdir = -self.gdir
                self.vy = -0.5 * g * C.GRAVITY_BALL  # лёгкий толчок от поверхности
        elif m == C.MODE_SPIDER:
            self.vy -= g * C.GRAVITY_SPIDER
            if press_edge:
                self._spider_teleport(size)

        self.vy = float(np.clip(self.vy, -C.MAX_FALL, C.MAX_FALL))

    def _spider_teleport(self, size):
        """Мгновенный телепорт на противоположную поверхность с флипом гравитации."""
        self.gdir = -self.gdir
        cols = self._cols(size) or [int(np.floor(self.x))]
        if self.gdir == -1:
            # летим к потолку: ищем ближайший блок сверху
            target = C.LEVEL_HEIGHT - size
            start = int(np.floor(self.y + size))
            for col in cols:
                for row in range(max(start, 0), C.LEVEL_HEIGHT):
                    if self.grid[row, col] == C.BLOCK:
                        target = min(target, row - size)
                        break
            self.y = target
        else:
            # летим к полу: ищем ближайший блок снизу
            target = 0.0
            start = min(int(np.floor(self.y)), C.LEVEL_HEIGHT - 1)
            for col in cols:
                for row in range(start, -1, -1):
                    if self.grid[row, col] == C.BLOCK:
                        target = max(target, row + 1.0)
                        break
            self.y = target
        self.vy = 0.0

    # -------------------------------------------------------- столкновения

    def _resolve_tiles(self, prev_y, size):
        eps = 1e-6
        cols = self._cols(size)
        r0 = max(0, int(np.floor(self.y + 0.02)))
        r1 = min(C.LEVEL_HEIGHT - 1, int(np.floor(self.y + size - 0.02)))

        solids = []
        for col in cols:
            for row in range(r0, r1 + 1):
                t = self.grid[row, col]
                if t == C.BLOCK:
                    solids.append(row)
                elif t == C.FINISH:
                    self.won = True
                elif t == C.SPIKE:
                    if self._hazard_hit(row, col, size):
                        self.dead = True
                elif t in C.PORTAL_TO_MODE:
                    if self.mode != C.PORTAL_TO_MODE[t]:
                        self.mode = C.PORTAL_TO_MODE[t]
                        self.hold_ticks = 0
                elif t in C.SPEED_VALUES:
                    self.speed_portal = t
                elif t == C.SIZE_MINI:
                    self.mini = True
                elif t == C.SIZE_NORMAL:
                    self.mini = False
                elif t == C.GRAV_FLIP:
                    self.gdir = -1
                elif t == C.GRAV_NORMAL:
                    self.gdir = 1

        if not solids or self.dead:
            return
        if self.mode == C.MODE_WAVE:
            self.dead = True   # волна умирает от любого касания блока
            return

        tops = [r + 1.0 for r in solids]
        bottoms = [float(r) for r in solids]
        if self.vy <= 0 and prev_y >= max(tops) - 0.15:
            # приземление сверху
            self.y = max(tops)
            self.vy = 0.0
        elif self.vy >= 0 and prev_y + size <= min(bottoms) + 0.15:
            # удар о нижнюю грань блока (потолок)
            self.y = min(bottoms) - size
            self.vy = 0.0 if self.gdir == 1 else self.vy
            if self.vy > 0:
                self.vy = 0.0
        else:
            self.dead = True   # боковое столкновение

    def _hazard_hit(self, row, col, size):
        """Шипы имеют уменьшенный хитбокс — касание краем прощается."""
        s = C.HAZARD_SHRINK
        hx0, hx1 = col + s, col + 1 - s
        hy0, hy1 = row + s, row + 1 - s
        px0, px1 = self.x - size / 2, self.x + size / 2
        py0, py1 = self.y, self.y + size
        return px0 < hx1 and px1 > hx0 and py0 < hy1 and py1 > hy0

    # ---------------------------------------------------------- наблюдение

    def _observe(self):
        H, W = C.OBS_H, C.OBS_W
        window = np.zeros((C.OBS_CHANNELS, H, W), dtype=np.float32)
        c0 = int(np.floor(self.x)) - C.OBS_BACK
        lo = max(c0, 0)
        hi = min(c0 + W, self.width)
        if hi > lo:
            patch = self.grid[:, lo:hi]                    # [H, w]
            dst = slice(lo - c0, hi - c0)
            window[0, :, dst] = (patch == C.BLOCK)
            window[1, :, dst] = (patch == C.SPIKE)
            window[2, :, dst] = (patch == C.DECOR)
            portal = (patch >= C.FINISH) & (patch != C.DECOR) & \
                     (patch != C.BLOCK) & (patch != C.SPIKE)
            window[3, :, dst] = portal
            window[4, :, dst] = patch * portal / 50.0      # код портала
        # окно хранится снизу вверх — переворачивать не нужно, сеть безразлична

        size = self._size()
        state = np.zeros(C.N_STATE, dtype=np.float32)
        state[0] = self.y / C.LEVEL_HEIGHT
        state[1] = self.vy / C.MAX_FALL
        state[2] = self.gdir
        state[3] = 1.0 if self._check_support(size) else 0.0
        state[4 + self.mode] = 1.0                          # one-hot режима (7)
        state[11] = C.SPEED_VALUES[self.speed_portal] / 0.32
        state[12] = 1.0 if self.mini else 0.0
        state[13] = float(self.prev_action)
        state[14] = self.x - np.floor(self.x)
        state[15] = self.x / self.width
        return np.concatenate([window.ravel(), state])

    # ------------------------------------------------------------- рендер

    def render(self, view_w=60):
        """ASCII-кадр уровня вокруг игрока."""
        chars = {
            C.EMPTY: " ", C.BLOCK: "#", C.SPIKE: "^", C.DECOR: "·",
            C.FINISH: "|",
        }
        c0 = max(0, int(self.x) - 6)
        c1 = min(self.width, c0 + view_w)
        rows = []
        prow = int(np.clip(np.floor(self.y + self._size() / 2), 0, C.LEVEL_HEIGHT - 1))
        pcol = int(np.floor(self.x))
        player_char = "@" if not self.mini else "o"
        for row in range(C.LEVEL_HEIGHT - 1, -1, -1):
            line = []
            for col in range(c0, c1):
                t = self.grid[row, col]
                if row == prow and col == pcol:
                    line.append(player_char)
                elif t in chars:
                    line.append(chars[t])
                elif t in C.PORTAL_TO_MODE:
                    line.append("M")
                elif t in C.SPEED_VALUES:
                    line.append("S")
                elif t in C.SIZE_PORTALS:
                    line.append("Z")
                else:
                    line.append("G")
            rows.append("".join(line))
        rows.append("=" * (c1 - c0))
        hud = (f"mode={C.MODE_NAMES[self.mode]} speed={C.SPEED_NAMES[self.speed_portal]} "
               f"{'mini ' if self.mini else ''}{'grav↑ ' if self.gdir == -1 else ''}"
               f"x={self.x:.1f}/{self.width} ({100 * self.x / self.width:.0f}%)")
        return "\n".join(rows) + "\n" + hud


class VecEnv:
    """Простейший векторизованный набор сред с авто-сбросом."""

    def __init__(self, n, config_fn, seed=0):
        self.envs = [GDEnv(config_fn(), seed=seed + i) for i in range(n)]
        self.n = n
        self.config_fn = config_fn

    def set_config(self, config_fn):
        self.config_fn = config_fn

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, d, i = env.step(a)
            if d:
                env.cfg = self.config_fn()   # новый случайный уровень каждый эпизод
                o = env.reset()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(i)
        return np.stack(obs), np.array(rews, dtype=np.float32), \
            np.array(dones, dtype=bool), infos
