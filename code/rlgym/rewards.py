from typing import List, Dict, Any, Tuple, Optional
from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
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
    
class KickoffReward(RewardFunction[AgentID, GameState, float]):
    """Récompense l'agent pour aller vers la balle pendant le kickoff,
    avec un bonus optionnel pour le premier contact."""

    def __init__(self, first_touch_weight: float = 0.0):
        self.vel_dir_reward = VelocityPlayerToBallReward()
        self.first_touch_weight = first_touch_weight

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        self.vel_dir_reward.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState,
                is_terminated: Dict[AgentID, bool],
                is_truncated: Dict[AgentID, bool],
                shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        is_kickoff = np.linalg.norm(state.ball.position[:2]) < 1.0

        if not is_kickoff:
            return {agent: 0.0 for agent in agents}

        rewards = self.vel_dir_reward.get_rewards(agents, state, is_terminated, is_truncated, shared_info)

        if self.first_touch_weight > 0.0:
            # Vérifie si quelqu'un touche la balle ce step et si personne n'a encore été récompensé
            touched_agent = None
            for agent in agents:
                if state.cars[agent].ball_touches > 0:
                    touched_agent = agent
                    break

            if touched_agent is not None and not any(self._touch_rewarded.values()):
                for agent in agents:
                    self._touch_rewarded[agent] = True
                    if agent == touched_agent:
                        rewards[agent] += self.first_touch_weight
                    else:
                        rewards[agent] -= self.first_touch_weight

        return rewards
    

    
class CloseRangeFaceBallReward(RewardFunction[AgentID, GameState, float]):
    """Forces the bot to turn and face the ball only when preparing to strike."""

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            pos_diff = state.ball.position - car.physics.position
            dist = float(np.linalg.norm(pos_diff))

            if dist < 1000.0:
                norm_pos_diff = pos_diff / dist if dist > 0 else pos_diff
                alignment = float(np.dot(car.physics.rotation_mtx[:, 0], norm_pos_diff))
                rewards[agent] = alignment
            else:
                rewards[agent] = 0.0

        return rewards
    
class EnergyReward(RewardFunction[AgentID, GameState, float]):
    """
    Récompense l'agent pour maintenir un haut niveau d'énergie mécanique totale.
    Combine hauteur, vitesse, boost, et flip/jump disponible en un signal unifié.
    """

    CAR_MASS = common_values.CAR_MASS  # masse approximative en rlgym

    JUMP_VEL = 292.0

    PER_BOOST_POTENTIAL = (0.5 * CAR_MASS * 3000.0 ** 2) / 100.0
    JUMP_POTENTIAL = 0.5 * CAR_MASS * JUMP_VEL ** 2 * 4 # equivalent à faire (JUMP_VEL * 2)^2
    MAX_ENERGY = (
        100 * PER_BOOST_POTENTIAL
        + JUMP_POTENTIAL
        + (CAR_MASS * common_values.GRAVITY * (common_values.CEILING_Z - 17))
        + (0.5 * CAR_MASS * common_values.CAR_MAX_SPEED ** 2)
    )

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        for agent in agents:
            car = state.cars[agent]

            if car.is_demoed:
                rewards[agent] = 0.0
                continue

            height = car.physics.position[2]
            velocity = float(np.linalg.norm(car.physics.linear_velocity))

            energy = 0.0

            # Énergie potentielle gravitationnelle (surpondérée x1.1 pour encourager le jeu aérien)
            energy += 1.1 * self.CAR_MASS * common_values.GRAVITY * height

            # Énergie cinétique
            energy += 0.5 * self.CAR_MASS * velocity ** 2

            # Énergie stockée dans le boost
            energy += self.PER_BOOST_POTENTIAL * car.boost_amount

            # Énergie potentielle du saut (si pas encore sauté)
            if not car.has_jumped:
                energy += self.JUMP_POTENTIAL

            # Énergie potentielle du flip/dodge
            if car.has_flip:
                dodge_impulse = (500.0 + velocity / 17.0) if velocity <= 1700.0 else (600.0 - (velocity - 1700.0))
                dodge_impulse = max(dodge_impulse - 25.0, 0.0)
                energy += 0.9 * 0.5 * self.CAR_MASS * dodge_impulse ** 2

            rewards[agent] = energy / self.MAX_ENERGY

        return rewards