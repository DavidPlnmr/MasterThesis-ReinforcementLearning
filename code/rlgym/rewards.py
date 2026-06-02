from typing import List, Dict, Any, Tuple, Optional
from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np

class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}
    
class FaceBallReward(RewardFunction):
    """Rewards the agent for facing the ball"""
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass


    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            ball = state.ball

            car_pos = car.physics.position
            ball_pos = ball.position
            direction_to_ball = ball_pos - car_pos
            norm = np.linalg.norm(direction_to_ball)

            if norm > 0:
                direction_to_ball /= norm

            car_forward = car.physics.forward
            dot_product = np.dot(car_forward, direction_to_ball)

            reward = dot_product  # Dot product directly indicates alignment (-1 to 1)
            rewards[agent] = reward

        return rewards

class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the opponent's goal"""
    
    def __init__(self, zero_sum: bool = False):
        self.zero_sum = zero_sum

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist
            
            vel_toward_goal = np.dot(ball_vel, dir_to_goal)

            if self.zero_sum:
                rewards[agent] = vel_toward_goal / common_values.BALL_MAX_SPEED
            else:
                rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0)
            
        return rewards

class ZeroSumReward(RewardFunction[AgentID, GameState, float]):
    '''
    child_reward: The underlying reward function
    team_spirit: How much to share this reward with teammates (0-1)
    opp_scale: How to scale the penalty we get for the opponents getting this reward (usually 1)
    '''

    def __init__(self, child_reward: RewardFunction, team_spirit: float, opp_scale: float = 1.0):
        self.child_reward = child_reward
        self.team_spirit = team_spirit
        self.opp_scale = opp_scale

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.child_reward.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        
        child_rewards = self.child_reward.get_rewards(agents, state, is_terminated, is_truncated, shared_info)

        
        team_reward_lists = [[], []]
        for agent in agents:
            team = int(state.cars[agent].is_orange)  # 0 = blue, 1 = orange
            team_reward_lists[team].append(child_rewards[agent])

        
        for i in range(2):
            if len(team_reward_lists[i]) == 0:
                team_reward_lists[i].append(0)

        team_rewards = [np.mean(team_reward_lists[i]) for i in range(2)]

        
        final_rewards = {}
        for agent in agents:
            team = int(state.cars[agent].is_orange)
            opp_team = 1 - team
            final_rewards[agent] = (
                child_rewards[agent] * (1 - self.team_spirit)
                + team_rewards[team] * self.team_spirit
                - team_rewards[opp_team] * self.opp_scale
            )

        return final_rewards

class LogCombinedReward(RewardFunction[AgentID, GameState, float]):
    def __init__(
        self,
        reward_functions: Tuple[RewardFunction, ...],
        reward_weights: Optional[Tuple[float, ...]] = None
    ):
        self.reward_functions = reward_functions
        self.reward_weights = reward_weights or np.ones(len(reward_functions))
        self.prev_rewards = None  # Shape: (n_rewards,) — moyenne sur les agents du dernier step

        if len(self.reward_functions) != len(self.reward_weights):
            raise ValueError("Reward functions and weights must have the same length")

    @classmethod
    def from_zipped(cls, *rewards_and_weights):
        rewards, weights = [], []
        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.0
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, agents, initial_state, shared_info):
        for func in self.reward_functions:
            func.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        # Récupère les rewards de chaque fonction pour tous les agents
        all_rewards = [
            func.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for func in self.reward_functions
        ]

        # Stocke la moyenne par reward function pour le logging
        self.prev_rewards = np.array([
            np.mean([r[agent] for agent in agents])
            for r in all_rewards
        ])

        # Combine les rewards pondérées
        final_rewards = {}
        for agent in agents:
            final_rewards[agent] = float(sum(
                w * r[agent]
                for w, r in zip(self.reward_weights, all_rewards)
            ))

        return final_rewards
