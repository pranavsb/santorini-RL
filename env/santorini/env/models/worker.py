"""
Represents a worker piece. Board handles this class (composition).
"""
from typing import Set, Tuple
import random


class Worker:
    def __init__(self, player_id: int, worker_id: int, occupied: Set[Tuple[int, int]]):
        self.player_id = player_id
        self.worker_id = worker_id
        random_location = Worker._get_random_board_location()
        while random_location in occupied:
            random_location = Worker._get_random_board_location()
        self.location = random_location
        occupied.add(random_location)

    @staticmethod
    def _get_random_board_location():
        return tuple((random.randint(0, 4), random.randint(0, 4)))
