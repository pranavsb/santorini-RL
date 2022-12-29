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
from typing import Optional

from pettingzoo.utils.env import AgentID, ActionType
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from models.board import Board


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
        # generate board with randomly placed worker pieces
        self.board = Board()

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

        self.rewards = {i: 0 for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.terminations = {i: False for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode

    def step(self, action: ActionType) -> None:
        return
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        current_agent = self.agent_selection
        # chosen_move = action_to_move(action)
        # assert self.board.is_legal_move(action)
        # TODO how to handle if agent plays when its not its turn?
        assert self.board.is_legal_action(action, self.agents.index(current_agent))
        self.board.move_and_build(action, self.agents.index(current_agent))

        # TODO test workers trapped by checking no legal moves left
        # TODO create action mask by checking all possible legal moves and see how to update self.obs.action_mask

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> None:
        # reset game state
        self.board.reset()

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
        printable_board = self.board.generate_printable_board()

        for printable_row in printable_board:
            print(printable_row)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]


if __name__ == "__main__":
    santorini_env = env()
    santorini_env.reset()
    santorini_env.render()
