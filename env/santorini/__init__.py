from gym.envs.registration import register

register(
    id='santorini-v0',
    entry_point='santorini.env:Santorini',
)