"""
Hold all game state in Board. Handles Santorini game logic as well.

Handles the Worker pieces and keeps track of their positions and building status of every square on the grid.
Also provides helper utilities for env.
"""

from typing import List, Dict, Tuple
from worker import Worker
import gymnasium
from util import *
import numpy as np
from numpy.typing import ArrayLike


class Board:
    def __init__(self):
        # keeps track of board height and buildings in 5x5 grid
        self.board_height: ArrayLike = np.zeros((5, 5), dtype=np.int8)

        # worker occupied locations
        self.occupied_locations = set()

        # keeps track of both workers of both players
        self.workers: Dict[int, List['Worker']] = {
            0: [Worker(player_id=0, worker_id=0, occupied=self.occupied_locations),
                             Worker(player_id=0, worker_id=1, occupied=self.occupied_locations)],
            1: [Worker(player_id=1, worker_id=0, occupied=self.occupied_locations),
                                 Worker(player_id=1, worker_id=1, occupied=self.occupied_locations)]
        }  # randomize worker placement for now, later can train RL to do it

        self.winner_decided = False  # just for sanity check assertions

    def is_legal_action(self, action: int, player_id: int) -> bool:
        worker_id, move_direction, build_direction = action_to_move(action)
        worker = self.workers[player_id][worker_id]
        move_coordinate = direction_to_coordinate(move_direction)
        move_to_coordinate = (worker.location[0] + move_coordinate[0], worker.location[1] + move_coordinate[1])
        if not self._can_move(worker.location, move_to_coordinate):
            # worker could not be moved to the specified move location
            return False
        if self.board_height[move_to_coordinate[0]][move_to_coordinate[1]] == 3:
            # game over, this is a valid jump to level 3, the build input doesn't matter
            return True
        build_coordinate = direction_to_coordinate(build_direction)
        build_to_coordinate = (move_to_coordinate[0] + build_coordinate[0], move_to_coordinate[1] + build_coordinate[1])
        if not self._can_build(worker.location, build_to_coordinate):
            # worker could move but not build at specified location
            return False
        return True

    def any_legal_moves(self, player_id: int) -> bool:
        return len(self.get_legal_moves(player_id, any_legal=True)) > 0

    def get_legal_moves(self, player_id: int, any_legal: bool = False) -> List[int]:
        if self.winner_decided:
            gymnasium.logger.warn("Getting legal moves after game winner decided.")
        workers = self.workers[player_id]
        legal_moves = []
        for worker_id in range(2):
            for move_direction in range(8):
                for build_direction in range(8):
                    action = move_to_action(tuple((worker_id, move_direction, build_direction)))
                    if self.is_legal_action(action, player_id):
                        legal_moves.append(action)
                        if any_legal:
                            # found one legal move, we can return early
                            return legal_moves
        return legal_moves

    def has_won(self):
        # returns player_id if any worker is present on level 3 or opponents workers trapped
        for player_id in range(2):
            for worker in self.workers[player_id]:
                x, y = worker.location[0], worker.location[1]
                assert self.board_height[x][y] < 4, "how is worker standing on a dome?"
                if self.board_height[x][y] == 3:
                    gymnasium.logger.info('winner decided {} by board height {},{}'.format(player_id, x, y))
                    for row in self.generate_printable_board():
                        gymnasium.logger.info("".join(row))
                    self.winner_decided = True
                    return player_id
            if not self.any_legal_moves(player_id):
                gymnasium.logger.info('winner decided {} by no opponent legal moves'.format(1-player_id))
                self.winner_decided = True
                return 1 - player_id
        return -1

    def _can_move(self, from_coordinate: Tuple[int, int], to_coordinate: Tuple[int, int]) -> bool:
        return within_grid_bounds(to_coordinate) and self._height_jump_valid(from_coordinate, to_coordinate) and not self._is_occupied(to_coordinate)

    def _can_build(self, start_coordinate: Tuple[int, int], to_coordinate: Tuple[int, int]) -> bool:
        if not within_grid_bounds(to_coordinate):
            return False
        # make sure that if it is occupied, it is due to the same worker piece that's being played
        not_occupied = start_coordinate == to_coordinate or not self._is_occupied(to_coordinate)
        # make sure dome doesn't already exist
        return not_occupied and self.board_height[to_coordinate[0]][to_coordinate[1]] < 4

    def generate_printable_board(self) -> List[List[str]]:
        printable_board = [["" for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                worker_string = self._get_worker_string(i, j)
                board_string = self._get_board_position_string(i, j, worker_string)
                printable_board[i][j] = pad_width(board_string, 10)

        return printable_board

    def _get_worker_string(self, x: int, y: int) -> str:
        for worker_id, worker in enumerate(self.workers[0]):
            if worker.location == (x, y):
                return "P0W{}".format(worker_id)
        for worker_id, worker in enumerate(self.workers[1]):
            if worker.location == (x, y):
                return "P1W{}".format(worker_id)
        return ""

    def _get_board_position_string(self, x: int, y: int, worker_string: str) -> str:
        if self.board_height[x][y] == 4:
            return "[[[ O ]]]"
        return ("[" * self.board_height[x][y]) + worker_string + ("]" * self.board_height[x][y])

    def reset(self):
        self.board_height: ArrayLike = np.zeros((5, 5), dtype=np.int8)
        self.occupied_locations = set()
        # randomize worker placement for now, later can train RL to do it
        self.workers: Dict[int, List['Worker']] = {
            0: [Worker(player_id=0, worker_id=0, occupied=self.occupied_locations),
                Worker(player_id=0, worker_id=1, occupied=self.occupied_locations)],
            1: [Worker(player_id=1, worker_id=0, occupied=self.occupied_locations),
                Worker(player_id=1, worker_id=1, occupied=self.occupied_locations)]
        }
        self.winner_decided = False

    def move_and_build(self, action: int, player_id: int) -> None:
        assert not self.winner_decided, "why move is played after winner decided?"
        worker_id, move_direction, build_direction = action_to_move(action)
        # update worker location
        move_coordinate = direction_to_coordinate(move_direction)
        worker = self.workers[player_id][worker_id]
        gymnasium.logger.debug("action: P{}W{} move {} build {} worker at:({}, {})".format(player_id, worker_id,
              move_direction, build_direction, worker.location[0], worker.location[1]))
        worker.location = (worker.location[0] + move_coordinate[0], worker.location[1] + move_coordinate[1])

        # if moved to level 3, game is over
        assert self.board_height[worker.location[0]][worker.location[1]] <= 3, "how did worker move to dome?"
        if self.board_height[worker.location[0]][worker.location[1]] == 3:
            return

        # build on the board
        build_coordinate = direction_to_coordinate(build_direction)
        build_location = (worker.location[0] + build_coordinate[0], worker.location[1] + build_coordinate[1])
        # # if build is out of bounds, the move is a winning one
        # if within_grid_bounds(build_location):
        self.board_height[build_location[0]][build_location[1]] += 1

    def _height_jump_valid(self, from_coordinate: Tuple[int, int], to_coordinate: Tuple[int, int]) -> bool:
        return self.board_height[from_coordinate[0]][from_coordinate[1]] + 1 >= self.board_height[to_coordinate[0]][to_coordinate[1]]

    def _is_occupied(self, coordinate: Tuple[int, int]) -> bool:
        # worker is present
        for player_id in self.workers:
            for worker in self.workers[player_id]:
                if worker.location == coordinate:
                    return True
        # dome is present
        return self.board_height[coordinate[0]][coordinate[1]] == 4

    def _worker_grid_nparray(self, player_id: int):
        worker_grid = np.zeros((5, 5), dtype=np.int8)
        for worker in self.workers[player_id]:
            worker_grid[worker.location[0]][worker.location[1]] = 1
        return worker_grid

    def get_observation(self, player_id: int) -> ArrayLike:
        self_workers = self._worker_grid_nparray(player_id)
        opponent_workers = self._worker_grid_nparray(1 - player_id)
        board_height = self.board_height
        observation_space = np.zeros((3, 5, 5), dtype=np.int8)
        np.stack([self_workers, opponent_workers, board_height], axis=0, out=observation_space)
        return observation_space
