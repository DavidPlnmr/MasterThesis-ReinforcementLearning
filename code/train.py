# This file is just a basic test script.
# For the usage of gymnasium environments, please refer to the documentation:
# https://gymnasium.farama.org

import gymnasium as gym

from stable_baselines3 import PPO

# Initialise the environment
environment_name = "CarRacing-v3"

env = gym.make(environment_name, render_mode="rgb_array") 

# I don't really know what is CnnPolicy and PPO. I followed an example
# I know that the policy is the decision-making part of the agent.
# That defines how the agent selects actions based on observations from the environment.
# Didn't check about that but Cnn policy might be linked to convolutional neural networks (CNNs) used 
# to process visual input.

model = PPO("CnnPolicy", env, verbose=1) 
model.learn(total_timesteps=200_000) 
model.save("ppo_" + environment_name.lower())

