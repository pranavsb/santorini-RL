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


    Action is 2 * 8 * 8 which represents choice of worker piece, direction to move and direction to build.
    This is represented as a Discrete(128) action space.

    Observation consists of three 5x5 planes. The first 5x5 plane is 1 for the agent's worker pieces and 0 otherwise.
    The second 5x5 plane is 1 for the opponent's worker pieces and 0 otherwise. The third 5x5 plane represents the height
    of the board at a given cell in the grid - this ranges from 0 (no buildings) to 4 (dome).

    Reward is 10 for winning, -10 for losing and -0.1 for every time step.

"""
import gymnasium
import numpy as np
from typing import Optional

from pettingzoo.utils.env import AgentID, ActionType
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from models.board import Board
from util import agent_id_to_name


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
        current_player_id = self.agents.index(current_agent)
        # chosen_move = action_to_move(action)
        # assert self.board.is_legal_move(action)
        # TODO how to handle if agent plays when its not its turn?
        assert self.board.is_legal_action(action, current_player_id)
        self.board.move_and_build(action, current_player_id)

        # TODO test workers trapped by checking no legal moves left
        opponent_has_legal_move = self.board.any_legal_moves()
        # note that credit assignment is complicated as move+build is one action and the winning move is a "move" where
        # the build is inconsequential, hopefully it shouldn't be a big deal
        current_player_won = self.board.has_won()  # only current player's worker can be on level 3 since its their turn
        # TODO create action mask by checking all possible legal moves and see how to update self.obs.action_mask

        game_over = not opponent_has_legal_move or current_player_won
        if game_over:
            self.set_game_result(current_player_id, reward_scaling_factor=10)
        else:
            # small negative reward to incentivize faster game completion
            self.rewards[current_agent] = -0.1

        self.agent_selection = (
            self._agent_selector.next()
        )  # Give turn to the next agent

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> None:
        self.has_reset = True
        self.agents = self.possible_agents[:]
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

    def set_game_result(self, winning_player_id: int, reward_scaling_factor=1):
        reward = reward_scaling_factor
        self.rewards[agent_id_to_name(1)] = reward if winning_player_id == 0 else -reward
        self.rewards[agent_id_to_name(2)] = reward if winning_player_id == 1 else -reward
        for i in self.agents:
            self.terminations[i] = True
            self.infos[i] = {"legal_moves": []}

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
