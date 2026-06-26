from rlgym_sac.util import RLGymV2GymWrapper  # <-- swap: rlgym_ppo.util -> 
import numpy as np
import gymnasium as gym

class FixedRLGymV2GymWrapper(RLGymV2GymWrapper):
    def __init__(self, rlgym_env):
        super().__init__(rlgym_env)
        if self.action_space is None and not self.is_discrete:
            n = list(rlgym_env.action_spaces.values())[0][0]
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(int(n),), dtype=np.float32
            )