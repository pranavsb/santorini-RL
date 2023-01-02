# Source: https://tianshou.readthedocs.io/en/latest/tutorials/tictactoe.html#two-random-agents
import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from env.santorini.env.santorini import env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


santorini_env = PettingZooEnv(env(render_mode="human"))

obs = santorini_env.reset()
santorini_env.render()
policy = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], santorini_env)
vector_env = DummyVectorEnv([lambda: santorini_env])
collector = Collector(policy, vector_env)
result = collector.collect(n_episode=1, render=.1)
