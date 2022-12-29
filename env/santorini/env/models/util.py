from enum import Enum


class BoardHeight(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class BoardUtils:
    @staticmethod
    def pad_width(string: str, pad_length: int) -> str:
        padding = pad_length - len(string)
        string = " " * (padding // 2) + string + " " * (padding // 2)
        if padding % 2 == 1:
            string += " "
        return string
