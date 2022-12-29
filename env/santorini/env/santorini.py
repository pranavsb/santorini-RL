"""
    Reuseable PettingZoo environment (similar to OpenAI Gym) for the Santorini board game.
    Official rules can be found at: https://roxley.com/products/santorini

    This is a classic environment since Santorini is a turn-based board game.
    Only ASCII env render is supported and 2 players (agents) play against each other.

    Note that we currently have three modifications to the official game rules:
    1. We randomly place the worker pieces at the start of the game.
    2. We don't support God powers.
    3. We have an unlimited number of building pieces.

    None of the modifications should have a significant effect on gameplay or favor a particular player.

    |--------------------|-----------------------------------------------|
    | Actions            | Discrete                                      |
    | Parallel API       | No                                            |
    | Manual Control     | No      # TODO                                |
    | Agents             | `agents= ['player_1', 'player_2']`            |
    | Agents             | 2                                             |
    | Action Shape       | (1)                                           |
    | Action Values      | [0, 127]                                      |
    | Observation Shape  | (5, 5, 3)                                     |
    | Observation Values | [0, 4]                                        |



"""
import gymnasium
import numpy as np
from typing import Optional, List

from pettingzoo.utils.env import AgentID, ActionType

from util import BoardUtils, Worker
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers



def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "santorini_v1",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        # todo move to board, and move worker to new worker? move both to /models ?
        self.board = [[0 for _ in range(5)] for _ in range(5)]

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(2 * 8 * 8) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=4, shape=(5, 5, 3), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(128,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = None
        self.infos = {i: {} for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.terminations = {i: False for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode

        self.occupied_locations = set()
        # randomize worker placement for now, later can train RL to do it
        self.self_workers = [Worker(player_id=0, worker_id=0, occupied=self.occupied_locations),
                             Worker(player_id=0, worker_id=1, occupied=self.occupied_locations)]
        self.opponent_workers = [Worker(player_id=1, worker_id=0, occupied=self.occupied_locations),
                                 Worker(player_id=1, worker_id=1, occupied=self.occupied_locations)]

    def step(self, action: ActionType) -> None:
        pass

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> None:
        # reset game state
        self.board = [[0 for _ in range(5)] for _ in range(5)]
        self.occupied_locations = set()
        # randomize worker placement for now, later can train RL to do it
        self.self_workers = [Worker(player_id=0, worker_id=0, occupied=self.occupied_locations),
                             Worker(player_id=0, worker_id=1, occupied=self.occupied_locations)]
        self.opponent_workers = [Worker(player_id=1, worker_id=0, occupied=self.occupied_locations),
                                 Worker(player_id=1, worker_id=1, occupied=self.occupied_locations)]

        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    def render(self) -> None:
        """
        This is a classic environment so it will support only ASCII stdout as a render.
        :return: None, prints to stdout
        """
        printable_board = self._generate_printable_board()

        for i in range(5):
            print(printable_board[i])

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

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
    santorini_env = env()
    santorini_env.reset()
    santorini_env.render()
