from typing import List, Dict, Any
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
    
class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for facing the ball"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            pos_diff = state.ball.position - car.physics.position
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            rewards[agent] = float(np.dot(car.physics.forward, norm_pos_diff))
        return rewards

class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for the velocity of the ball towards the opponent's goal"""
    
    def __init__(self, own_goal=False):
        self.own_goal = own_goal

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            
            # Détermine l'objectif selon l'équipe
            if car.team_num == 0 and not self.own_goal or car.team_num == 1 and self.own_goal:
                objective = np.array([0, common_values.BACK_NET_Y, 0])   # but orange
            else:
                objective = np.array([0, -common_values.BACK_NET_Y, 0])  # but bleu

            vel = state.ball.linear_velocity
            pos_diff = objective - state.ball.position
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / common_values.BALL_MAX_SPEED

            rewards[agent] = float(np.dot(norm_pos_diff, norm_vel))
        
        return rewards