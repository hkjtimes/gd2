"""Процедурный генератор случайных уровней.

Уровень собирается из сегментов. Каждый сегмент строится под текущий режим
игрока (после портала режима паттерны меняются), с учётом текущей скорости
(чем быстрее — тем больше расстояния между препятствиями). Поверх игровой
геометрии рассыпаются декорации, которые ни на что не влияют.

Параметр difficulty (0..1) управляет плотностью препятствий и тем, какие
модификаторы разрешены. Это основа curriculum learning: сеть сначала учится
на простых уровнях, потом генератор подключает порталы скорости/размера/
режимов/гравитации.
"""

import numpy as np

from . import constants as C


class LevelConfig:
    """Что разрешено генератору. Управляется учебным планом (curriculum)."""

    def __init__(self, difficulty=0.5, allow_mode_portals=True,
                 allow_speed_portals=True, allow_size_portals=True,
                 allow_gravity_portals=True, decor_density=0.06,
                 min_columns=180, max_columns=320):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.allow_mode_portals = allow_mode_portals
        self.allow_speed_portals = allow_speed_portals
        self.allow_size_portals = allow_size_portals
        self.allow_gravity_portals = allow_gravity_portals
        self.decor_density = decor_density
        self.min_columns = min_columns
        self.max_columns = max_columns


def _gap_columns(speed_tile, jump_like=True):
    """Минимальный безопасный промежуток между препятствиями (в колонках)."""
    # За прыжок куба (~24 тика) игрок пролетает speed*24 тайлов;
    # оставляем запас, чтобы уровень оставался проходимым.
    horizon = speed_tile * 24.0
    base = int(np.ceil(horizon)) + (3 if jump_like else 2)
    return max(4, base)


class LevelGenerator:
    def __init__(self, config: LevelConfig, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng

    # ---------- публичный API ----------

    def generate(self):
        """Возвращает (grid, meta). grid: int8 [H, W] (y=0 — нижний ряд)."""
        cfg, rng = self.cfg, self.rng
        width = int(rng.integers(cfg.min_columns, cfg.max_columns + 1))
        grid = np.zeros((C.LEVEL_HEIGHT, width), dtype=np.int8)

        mode = C.MODE_CUBE
        speed_portal = C.SPEED_1X
        x = 0
        # Стартовая безопасная зона.
        x = self._flat(grid, x, 8)

        while x < width - 20:
            seg_len = int(rng.integers(14, 26))
            seg_len = min(seg_len, width - 20 - x)
            if seg_len <= 0:
                break
            speed = C.SPEED_VALUES[speed_portal]
            x = self._build_segment(grid, x, seg_len, mode, speed)
            # Между сегментами — шанс поставить портал-модификатор.
            x, mode, speed_portal = self._maybe_portal(grid, x, mode, speed_portal)

        # Финишная зона: пусто + финишная колонка.
        x = self._flat(grid, x, max(0, width - x - 2))
        grid[:, width - 1] = C.FINISH
        self._sprinkle_decor(grid)
        meta = {"width": width}
        return grid, meta

    # ---------- сегменты ----------

    def _flat(self, grid, x, n):
        return min(x + n, grid.shape[1])

    def _maybe_portal(self, grid, x, mode, speed_portal):
        cfg, rng = self.cfg, self.rng
        width = grid.shape[1]
        if x + 12 >= width:
            return x, mode, speed_portal

        choices = []
        if cfg.allow_mode_portals and cfg.difficulty > 0.15:
            choices += ["mode"] * 3
        if cfg.allow_speed_portals and cfg.difficulty > 0.25:
            choices += ["speed"] * 2
        if cfg.allow_size_portals and cfg.difficulty > 0.35:
            choices += ["size"]
        if cfg.allow_gravity_portals and cfg.difficulty > 0.45:
            choices += ["grav"]
        if not choices or rng.random() > 0.6:
            return x, mode, speed_portal

        kind = choices[int(rng.integers(len(choices)))]
        portal_col = x + 2
        if kind == "mode":
            new_mode = int(rng.integers(C.N_MODES))
            portal = C.MODE_PORTALS[new_mode]
            mode = new_mode
        elif kind == "speed":
            # На высокой сложности доступны более быстрые порталы.
            max_idx = 2 + int(cfg.difficulty * 2.999)  # до 4x при difficulty→1
            portal = C.SPEED_PORTALS[int(rng.integers(0, min(max_idx, 5)))]
            speed_portal = portal
        elif kind == "size":
            portal = C.SIZE_PORTALS[int(rng.integers(2))]
        else:
            portal = C.GRAV_PORTALS[int(rng.integers(2))]

        # Портал — вертикальная пара тайлов, чтобы его нельзя было промахнуться.
        # Гравитационные порталы опасны в полётных режимах — не переворачиваем
        # гравитацию там, где сверху потолок из блоков близко.
        grid[1:5, portal_col] = portal
        return portal_col + 4, mode, speed_portal

    def _build_segment(self, grid, x, seg_len, mode, speed):
        if mode in (C.MODE_CUBE, C.MODE_ROBOT):
            return self._segment_ground(grid, x, seg_len, speed)
        if mode in (C.MODE_SHIP, C.MODE_UFO):
            return self._segment_corridor(grid, x, seg_len, speed, min_gap=4)
        if mode == C.MODE_WAVE:
            return self._segment_corridor(grid, x, seg_len, speed, min_gap=3)
        # ball / spider: пол и потолок с шипами попеременно
        return self._segment_flip(grid, x, seg_len, speed)

    def _segment_ground(self, grid, x, seg_len, speed):
        """Наземный сегмент: шипы и невысокие ступеньки."""
        rng, cfg = self.rng, self.cfg
        end = x + seg_len
        gap = _gap_columns(speed, jump_like=True)
        density = 0.25 + 0.5 * cfg.difficulty
        cur = x + 2
        while cur < end - 3:
            if rng.random() < density:
                if rng.random() < 0.65:
                    # кластер шипов на земле, 1-3 подряд (уже — на скорости)
                    n = 1 + int(rng.integers(0, 2 + (cfg.difficulty > 0.5)))
                    n = min(n, max(1, int(2.5 - speed * 4)) + 1)
                    for i in range(n):
                        if cur + i < end:
                            grid[0, cur + i] = C.SPIKE
                    cur += n + gap
                else:
                    # ступенька из блоков высотой 1-2 (на неё можно запрыгнуть)
                    h = 1 + int(rng.random() < 0.35 * cfg.difficulty * 2)
                    w = int(rng.integers(2, 5))
                    for i in range(min(w, end - cur)):
                        grid[0:h, cur + i] = C.BLOCK
                    if rng.random() < 0.4 * cfg.difficulty and cur + 1 < end:
                        grid[h, cur + min(w, end - cur) - 1] = C.SPIKE
                    cur += w + gap
            else:
                cur += 1
        return end

    def _segment_corridor(self, grid, x, seg_len, speed, min_gap):
        """Коридор для полётных режимов: пол и потолок из блоков, зазор гуляет."""
        rng, cfg = self.rng, self.cfg
        end = x + seg_len
        H = C.LEVEL_HEIGHT
        gap_size = max(min_gap, int(round(H - 3 - cfg.difficulty * (H - 4 - min_gap))))
        center = H // 2
        max_shift = 1 if speed < 0.25 else 0  # на 3x-4x коридор не дёргаем
        for cx in range(x, end):
            center += int(rng.integers(-1, 2)) * (max_shift if cx % 3 == 0 else 0)
            center = int(np.clip(center, gap_size // 2 + 1, H - gap_size // 2 - 2))
            lo = center - gap_size // 2
            hi = center + (gap_size + 1) // 2
            grid[:max(0, lo), cx] = C.BLOCK
            grid[hi:, cx] = C.BLOCK
            # редкие шипы на стенках коридора
            if rng.random() < 0.10 * cfg.difficulty:
                if rng.random() < 0.5 and lo >= 1:
                    grid[lo, cx] = C.SPIKE
                elif hi < H - 1:
                    grid[hi - 1, cx] = C.SPIKE
        return end

    def _segment_flip(self, grid, x, seg_len, speed):
        """Сегмент для ball/spider: препятствия то на полу, то на потолке."""
        rng, cfg = self.rng, self.cfg
        end = x + seg_len
        H = C.LEVEL_HEIGHT
        # потолок из блоков, чтобы было к чему "прилипать"
        grid[H - 1, x:end] = C.BLOCK
        gap = _gap_columns(speed, jump_like=False)
        cur = x + 2
        top = False
        while cur < end - 2:
            if rng.random() < 0.3 + 0.4 * cfg.difficulty:
                n = 1 + int(rng.integers(0, 2))
                for i in range(n):
                    if cur + i >= end:
                        break
                    if top:
                        grid[H - 2, cur + i] = C.SPIKE
                    else:
                        grid[0, cur + i] = C.SPIKE
                top = not top
                cur += n + gap
            else:
                cur += 1
        return end

    # ---------- декорации ----------

    def _sprinkle_decor(self, grid):
        """Случайные декорации в пустых клетках. Игрок сквозь них пролетает."""
        rng = self.rng
        H, W = grid.shape
        mask = (grid == C.EMPTY) & (rng.random((H, W)) < self.cfg.decor_density)
        grid[mask] = C.DECOR
        # Иногда — плотные "колонны" декора, похожие на настоящие препятствия:
        # именно они заставляют сеть отличать декор от геометрии.
        for _ in range(int(W * self.cfg.decor_density * 0.6)):
            cx = int(rng.integers(0, W))
            h = int(rng.integers(2, 6))
            col = grid[:h, cx]
            col[col == C.EMPTY] = C.DECOR
