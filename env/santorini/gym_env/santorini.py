import gym
from typing import Tuple, TypeVar, Optional, Union, List

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class Santorini(gym.Env):
    def __init__(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass
