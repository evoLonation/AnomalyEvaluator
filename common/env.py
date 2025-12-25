import enum


class Env(enum.Enum):
    BXH = "bxh"
    ACT = "act"
    HANGZHOU = "hangzhou"

CURRENT_ENV = Env.HANGZHOU