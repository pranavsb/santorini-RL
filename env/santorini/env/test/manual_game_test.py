from env.santorini.env.models.board import Board
from env.santorini.env.models.util import *
import string

KEYBOARD_DIRECTION_TO_INDEX = {
    "w": 0,
    "e": 1,
    "d": 2,
    "c": 3,
    "x": 4,
    "z": 5,
    "a": 6,
    "q": 7,
}


def play_manually():
    current_player = 0
    while not board.has_won():
        print_board()
        move_input = input("Player {}: ".format(current_player))
        while invalid_input(move_input):
            move_input = input("Oops! Try again: ")
        action = keyboard_move_to_action(move_input)
        if not board.is_legal_action(action, current_player):
            continue
        board.move_and_build(action, current_player)
        current_player = (current_player + 1) % 2
    print("GAME OVER!")


def invalid_input(move):
    if len(move) != 3:
        return True
    if move[0] not in string.digits or int(move[0]) > 1:
        return True
    if move[1] not in KEYBOARD_DIRECTION_TO_INDEX or move[2] not in KEYBOARD_DIRECTION_TO_INDEX:
        return True
    return False


def keyboard_move_to_action(move):
    worker_id = int(move[0])
    move_direction = KEYBOARD_DIRECTION_TO_INDEX[move[1]]
    build_direction = KEYBOARD_DIRECTION_TO_INDEX[move[2]]
    action_id = move_to_action((worker_id, move_direction, build_direction))
    return action_id


def print_board():
    for row in board.generate_printable_board():
        print("\r" + str(row), end="\n")  # overwrite existing stdout line


if __name__ == "__main__":
    board = Board()
    play_manually()
