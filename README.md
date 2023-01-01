# Santorini RL
Play the board game Santorini with this Reinforcement Learning agent and custom Gym environment

This is a PettingZoo environment (similar to OpenAI Gym for multi-agent tasks) for the board game [Santorini](https://boardgamegeek.com/boardgame/194655/santorini).

Using this environment, different RL techniques like PPO and DQN to solve Santorini are attempted.

See `/env` for PettingZoo environment and `/model` for RL code and models.

Environment:

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
    | Manual Control     | No                                            |
    | Agents             | `agents= ['player_1', 'player_2']`            |
    | Agents             | 2                                             |
    | Action Shape       | (1)                                           |
    | Action Values      | [0, 127]                                      |
    | Observation Shape  | (3, 5, 5)                                     |
    | Observation Values | [0, 4]                                        |


    Action space is 2 * 8 * 8 which represents choice of worker piece, direction to move and direction to build.
    This is represented as a Discrete(128) action space. Creating a single discrete instead of MultiDiscrete space since
    that's what many standard implementations have done. For example, the Chess implementation by PettingZoo.

    Observation space consists of three 5x5 planes, represented as Box(3, 5, 5). The first 5x5 plane is 1 for the
    agent's worker pieces and 0 otherwise. The second 5x5 plane is 1 for the opponent's worker pieces and 0 otherwise.
    The third 5x5 plane represents the height of the board at a given cell in the grid - this ranges from 0 (no buildings) to 4 (dome).
    
    Reward is 10 for winning, -10 for losing and -0.1 for every time step.
