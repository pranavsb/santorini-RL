import gym
from typing import Tuple, TypeVar, Optional, Union, List
from util import BoardUtils, Worker

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class Santorini(gym.Env):
    def __init__(self):
        self.board = [[0 for _ in range(5)] for _ in range(5)]
        self.occupied_locations = set()
        # randomize worker placement for now, later can train RL to do it
        self.self_workers = [Worker(player_id=0, worker_id=0, occupied=self.occupied_locations),
                             Worker(player_id=0, worker_id=1, occupied=self.occupied_locations)]
        self.opponent_workers = [Worker(player_id=1, worker_id=0, occupied=self.occupied_locations),
                                 Worker(player_id=1, worker_id=1, occupied=self.occupied_locations)]

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        printable_board = self._generate_printable_board()

        for i in range(5):
            print(printable_board[i])

    def close(self):
        pass

    def _generate_printable_board(self) -> List[List[str]]:
        printable_board = [["" for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                worker_string = self._get_worker_string(i, j)
                board_string = self._get_board_position_string(i, j, worker_string)
                printable_board[i][j] = BoardUtils.pad_width(board_string, 10)

        return printable_board

    def _get_worker_string(self, x: int, y: int) -> str:
        for worker_id, worker in enumerate(self.self_workers):
            if worker.location == (x, y):
                return "P0W{}".format(worker_id)
        for worker_id, worker in enumerate(self.opponent_workers):
            if worker.location == (x, y):
                return "P1W{}".format(worker_id)
        return ""

    def _get_board_position_string(self, x: int, y: int, worker_string: str) -> str:
        if self.board[x][y] == 4:
            return "[[[ O ]]]"
        return "[" * self.board[x][y] + worker_string + "]" * self.board[x][y]


if __name__ == "__main__":
    env = Santorini()
    env.render()
