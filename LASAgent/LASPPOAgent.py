"""
Learning agent using PPO by OpenAI

"""

from baselines.ppo2.ppo2 import Model, Runner
from LASAgent.LASBaselineAgent import InternalEnvironment


class LASPPOAgent:

    def __init__(self, agent_name, observation_dim, action_dim, num_observation=20, env=None, env_type='Unity', load_pretrained_agent_flag=False, save_dir=None):
        self.ppo_agent = PPOAgent(agent_name, observation_dim, action_dim, env, env_type, load_pretrained_agent_flag, save_dir)
        self.internal_env = InternalEnvironment(observation_dim, action_dim, num_observation)


class PPOAgent:

    def __init__(self, agent_name, observation_dim, action_dim, env, env_type, load_pretrained_agent_flag, save_dir):
        self.agent_name = agent_name

