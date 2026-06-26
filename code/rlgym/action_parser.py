from typing import Dict, Any, List
import numpy as np
from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState


class ContinuousAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, int]):
    """
    Simple continuous action space that maps an array of 8 values on the interval [-1, 1] into an array of valid car
    controls.
    """
    def __init__(self):
        super().__init__()
        self._n_controller_inputs = 8

    def get_action_space(self, agent: AgentID) -> tuple:
        return float(self._n_controller_inputs), 'continuous'

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            car_controls = np.zeros(self._n_controller_inputs)
            car_controls[:] = action[:]
            car_controls[-3:] = np.round((car_controls[-3:] + 1) / 2)
            parsed_actions[agent] = car_controls
        return parsed_actions