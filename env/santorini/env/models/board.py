"""
Hold all game state in Board.

Handles the Worker pieces and keeps track of their positions and building status of every square on the grid.
Also provides helper utilities for env.
"""

from typing import List, Dict
from worker import Worker
from util import BoardUtils


class Board:
    def __init__(self):
        # keeps track of board height and buildings in 5x5 grid
        self.board_height: List[List[int]] = [[0 for _ in range(5)] for _ in range(5)]

        # worker occupied locations
        self.occupied_locations = set()

        # keeps track of both workers of both players
        self.workers: Dict[int, List['Worker']] = {
            1: [Worker(player_id=0, worker_id=0, occupied=self.occupied_locations),
                             Worker(player_id=0, worker_id=1, occupied=self.occupied_locations)],
            2: [Worker(player_id=1, worker_id=0, occupied=self.occupied_locations),
                                 Worker(player_id=1, worker_id=1, occupied=self.occupied_locations)]
        }  # randomize worker placement for now, later can train RL to do it

    def generate_printable_board(self) -> List[List[str]]:
        printable_board = [["" for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                worker_string = self._get_worker_string(i, j)
                board_string = self._get_board_position_string(i, j, worker_string)
                printable_board[i][j] = BoardUtils.pad_width(board_string, 10)

        return printable_board

    def _get_worker_string(self, x: int, y: int) -> str:
        for worker_id, worker in enumerate(self.workers[1]):
            if worker.location == (x, y):
                return "P0W{}".format(worker_id)
        for worker_id, worker in enumerate(self.workers[2]):
            if worker.location == (x, y):
                return "P1W{}".format(worker_id)
        return ""

    def _get_board_position_string(self, x: int, y: int, worker_string: str) -> str:
        if self.board_height[x][y] == 4:
            return "[[[ O ]]]"
        return "[" * self.board_height[x][y] + worker_string + "]" * self.board_height[x][y]

    def reset(self):
        self.board_height: List[List[int]] = [[0 for _ in range(5)] for _ in range(5)]
        self.occupied_locations = set()
        # randomize worker placement for now, later can train RL to do it
        self.workers: Dict[int, List['Worker']] = {
            1: [Worker(player_id=0, worker_id=0, occupied=self.occupied_locations),
                Worker(player_id=0, worker_id=1, occupied=self.occupied_locations)],
            2: [Worker(player_id=1, worker_id=0, occupied=self.occupied_locations),
                Worker(player_id=1, worker_id=1, occupied=self.occupied_locations)]
        }
