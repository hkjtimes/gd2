"""Константы игрового мира: тайлы, режимы игрока, скорости."""

# ---- Коды тайлов в сетке уровня ----
EMPTY = 0
BLOCK = 1          # твёрдый блок: сбоку — смерть, сверху — опора
SPIKE = 2          # шип: любое касание — смерть
DECOR = 3          # декорация: не влияет на игру, нейросеть должна её игнорировать
FINISH = 4         # финишная линия

# Порталы режимов (меняют режим игрока)
PORTAL_CUBE = 10
PORTAL_SHIP = 11
PORTAL_BALL = 12
PORTAL_UFO = 13
PORTAL_WAVE = 14
PORTAL_ROBOT = 15
PORTAL_SPIDER = 16

# Порталы скорости
SPEED_05X = 20
SPEED_1X = 21
SPEED_2X = 22
SPEED_3X = 23
SPEED_4X = 24

# Порталы размера
SIZE_MINI = 30
SIZE_NORMAL = 31

# Порталы гравитации
GRAV_FLIP = 40     # переворачивает гравитацию
GRAV_NORMAL = 41   # возвращает обычную гравитацию

MODE_PORTALS = (PORTAL_CUBE, PORTAL_SHIP, PORTAL_BALL, PORTAL_UFO,
                PORTAL_WAVE, PORTAL_ROBOT, PORTAL_SPIDER)
SPEED_PORTALS = (SPEED_05X, SPEED_1X, SPEED_2X, SPEED_3X, SPEED_4X)
SIZE_PORTALS = (SIZE_MINI, SIZE_NORMAL)
GRAV_PORTALS = (GRAV_FLIP, GRAV_NORMAL)
ALL_PORTALS = MODE_PORTALS + SPEED_PORTALS + SIZE_PORTALS + GRAV_PORTALS

# ---- Режимы игрока ----
MODE_CUBE = 0
MODE_SHIP = 1      # "ракета"
MODE_BALL = 2
MODE_UFO = 3
MODE_WAVE = 4
MODE_ROBOT = 5
MODE_SPIDER = 6
N_MODES = 7

MODE_NAMES = ("cube", "ship", "ball", "ufo", "wave", "robot", "spider")

PORTAL_TO_MODE = {
    PORTAL_CUBE: MODE_CUBE,
    PORTAL_SHIP: MODE_SHIP,
    PORTAL_BALL: MODE_BALL,
    PORTAL_UFO: MODE_UFO,
    PORTAL_WAVE: MODE_WAVE,
    PORTAL_ROBOT: MODE_ROBOT,
    PORTAL_SPIDER: MODE_SPIDER,
}

# Горизонтальная скорость, тайлов за тик (примерно по пропорциям GD)
SPEED_VALUES = {
    SPEED_05X: 0.139,
    SPEED_1X: 0.173,
    SPEED_2X: 0.216,
    SPEED_3X: 0.260,
    SPEED_4X: 0.320,
}
SPEED_NAMES = {
    SPEED_05X: "0.5x", SPEED_1X: "1x", SPEED_2X: "2x",
    SPEED_3X: "3x", SPEED_4X: "4x",
}

# ---- Геометрия мира ----
LEVEL_HEIGHT = 12          # высота уровня в тайлах (пол на y=0, потолок на y=LEVEL_HEIGHT)
PLAYER_SIZE = 0.8          # сторона хитбокса обычного игрока
PLAYER_SIZE_MINI = 0.44    # сторона хитбокса мини-игрока
HAZARD_SHRINK = 0.18       # шипы прощают касание краем (хитбокс шипа меньше тайла)

# ---- Физика (тайлы, тики; 1 тик ~ 1/60 c) ----
GRAVITY_CUBE = 0.028
JUMP_CUBE = 0.335
GRAVITY_ROBOT = 0.028
JUMP_ROBOT = 0.24
ROBOT_HOLD_BOOST = 0.014
ROBOT_MAX_HOLD = 12
GRAVITY_SHIP = 0.014
SHIP_THRUST = 0.030
SHIP_MAX_VY = 0.24
GRAVITY_UFO = 0.022
UFO_IMPULSE = 0.26
GRAVITY_BALL = 0.034
GRAVITY_SPIDER = 0.06
MAX_FALL = 0.5
MINI_JUMP_SCALE = 0.82     # мини-куб прыгает ниже

# ---- Наблюдение агента ----
OBS_BACK = 2               # сколько колонок видно позади игрока
OBS_AHEAD = 22             # сколько колонок видно впереди
OBS_W = OBS_BACK + OBS_AHEAD
OBS_H = LEVEL_HEIGHT
OBS_CHANNELS = 5           # solid, hazard, decor, portal_mask, portal_code
N_STATE = 16               # размер вектора состояния игрока
OBS_SIZE = OBS_W * OBS_H * OBS_CHANNELS + N_STATE

N_ACTIONS = 2              # 0 = отпустить, 1 = зажать (одна кнопка, как в GD)
