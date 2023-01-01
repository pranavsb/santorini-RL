# import stable_baselines3
# from supersuit import pettingzoo_env_to_vec_env_v1
import os
from ray import air, tune
from ray.tune.registry import register_env
from santorini.env.santorini import env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.apex_ddpg import ApexDDPGConfig
# from pettingzoo.utils.conversions import aec_to_parallel, turn_based_aec_to_parallel

if __name__ == "__main__":
    def env_creator(args):
        return PettingZooEnv(env())
    santorini_env = env_creator({})
    register_env("santorini", env_creator)

    config = (
        PPOConfig()
            .rollouts(num_rollout_workers=1, rollout_fragment_length=128)
            .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
            .environment(env="santorini", clip_actions=True)
            .debugging(log_level="INFO")
            .framework(framework="tf2")
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
            .multi_agent(
            policies=santorini_env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 10000},  #5000000},
        checkpoint_freq=10,
        local_dir="E:/projects/santorini-RL/ray_results/santorini",
        config=config.to_dict(),
    )

    # config = (
    #     ApexDDPGConfig()
    #         .environment("santorini")
    #         .resources(num_gpus=1)
    #         .rollouts(num_rollout_workers=1)
    #         .multi_agent(
    #         policies=santorini_env.get_agent_ids(),
    #         policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
    #     )
    # )
    #
    # tune.Tuner(
    #     "APEX_DDPG",
    #     run_config=air.RunConfig(
    #         stop={"episodes_total": 60000},
    #         checkpoint_config=air.CheckpointConfig(
    #             checkpoint_frequency=10,
    #         ),
    #     ),
    #     param_space=config,
    # ).fit()
