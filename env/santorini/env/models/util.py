from enum import Enum
from typing import Tuple


class TowerHeight(Enum):
    GROUND = 0
    ONE = 1
    TWO = 2
    THREE = 3
    DOME = 4


GRID_DIMENSION = 5


# counter-clockwise starting with North - N, NE, E, SE, S, SW, W, NW
INDEX_TO_DIRECTION = {
    0: "N",
    1: "NE",
    2: "E",
    3: "SE",
    4: "S",
    5: "SW",
    6: "W",
    7: "NW",
}

DIRECTION_TO_COORDINATE = {
    "N": (0, 1),
    "NE": (1, 1),
    "E": (1, 0),
    "SE": (1, -1),
    "S": (0, -1),
    "SW": (-1, -1),
    "W": (-1, 0),
    "NW": (-1, 1),
}


def pad_width(string: str, pad_length: int) -> str:
    padding = pad_length - len(string)
    string = " " * (padding // 2) + string + " " * (padding // 2)
    if padding % 2 == 1:
        string += " "
    return string


def within_grid_bounds(coordinate: Tuple[int, int]) -> bool:
    return 0 <= coordinate[0] < 5 and 0 <= coordinate[1] < 5


def action_to_move(action: int) -> Tuple[int, str, str]:
    assert 0 <= action < 128, "Action input is invalid."
    worker_id = action // 64
    action = action % 64

    assert 0 <= action < 64, "Action is invalid after extracting worker."
    move = action // 8
    move_direction = index_to_direction(move)
    action = action // 8

    assert 0 <= action < 8, "Action is invalid after extracting move direction."
    build = action % 8
    build_direction = index_to_direction(build)

    return worker_id, move_direction, build_direction


def move_to_action(move: Tuple[int, int, int]) -> int:
    assert 0 <= move[0] < 2, "Worker choice is invalid, can't convert move to action."
    action = 64 * move[0]

    assert 0 <= move[1] < 8, "Move direction invalid, can't convert move to action."
    action += (move[1] * 8)

    assert 0 <= move[2] < 8, "Build direction invalid, can't convert move to action."
    action += move[2]

    return action


def agent_id_to_name(agent_id: int) -> str:
    return f"player_{agent_id}"


def index_to_direction(move_build_index: int) -> str:
    assert 0 <= move_build_index < 8, "Invalid input while converting move/build index to direction."
    return INDEX_TO_DIRECTION[move_build_index]


def direction_to_coordinate(direction: str) -> Tuple[int, int]:
    assert direction in DIRECTION_TO_COORDINATE, "Invalid direction input, cannot convert to coordinate."
    return DIRECTION_TO_COORDINATE[direction]
