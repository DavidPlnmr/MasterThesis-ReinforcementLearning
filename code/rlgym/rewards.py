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
    """Rewards the agent for hitting the ball toward the opponent's goal (signal brut, non zero-sum)."""

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y

            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            dir_to_goal = pos_diff / dist
            vel_toward_goal = np.dot(ball.linear_velocity, dir_to_goal)
            rewards[agent] = vel_toward_goal / common_values.BALL_MAX_SPEED
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
    """
    Équivalent de CombinedReward (rlgym v2) qui mémorise la dernière
    contribution (pondérée) de chaque sous-reward dans self.prev_rewards,
    pour pouvoir les logger individuellement.
    """

    def __init__(self, *rewards_and_weights):
        self.reward_functions = []
        self.reward_names = []
        weights = []
        
        for value in rewards_and_weights:
            if isinstance(value, tuple):
                if len(value) == 3:
                    r, w, name = value
                elif len(value) == 2:
                    r, w = value
                    name = r.__class__.__name__
                else:
                    raise ValueError(f"Tuple doit avoir 2 ou 3 éléments, reçu {len(value)}")
            else:
                r, w, name = value, 1.0, value.__class__.__name__
            
            self.reward_functions.append(r)
            self.reward_names.append(name)
            weights.append(w)
        
        self.reward_weights = np.array(weights, dtype=np.float32)
        self.prev_rewards = np.zeros(len(self.reward_functions), dtype=np.float32)

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        for func in self.reward_functions:
            func.reset(agents, initial_state, shared_info)
        self.prev_rewards = np.zeros(len(self.reward_functions), dtype=np.float32)

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        # Chaque sous-reward renvoie un dict {agent: valeur}
        sub_rewards = [
            func.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for func in self.reward_functions
        ]

        # Pour le logging : on stocke la contribution pondérée moyennée sur les agents
        for i, rew_dict in enumerate(sub_rewards):
            self.prev_rewards[i] = self.reward_weights[i] * np.mean(
                [rew_dict[agent] for agent in agents]
            )

        # Reward combiné par agent
        combined = {}
        for agent in agents:
            total = 0.0
            for i, rew_dict in enumerate(sub_rewards):
                total += self.reward_weights[i] * rew_dict[agent]
            combined[agent] = total
        return combined