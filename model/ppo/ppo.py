# import stable_baselines3
from supersuit import pettingzoo_env_to_vec_env_v1
from santorini.env.santorini import env
# from pettingzoo.utils.conversions import aec_to_parallel, turn_based_aec_to_parallel

santorini_env = env()
# vec_env = pettingzoo_env_to_vec_env_v1(turn_based_aec_to_parallel(santorini_env))

# model = stable_baselines3.PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save("policy")

# model = stable_baselines3.PPO.load("policy")
# santorini_env = env()
# santorini_env.reset()
# for i in range(1000):
#     obs, reward, done, info = santorini_env.last(action)
#     act = model.predict(obs, deterministic=True)[0] if not done else None
#     santorini_env.step(act)
#     santorini_env.render()
#     if done:
#         santorini_env.reset()
