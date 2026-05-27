from typing import Dict, Any
import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values

from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_WALL_Y, CEILING_Z
from rlgym.rocket_league.math import rand_vec3, rand_uvec3, normalize
from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import RAMP_HEIGHT

from rlgym_tools.rocket_league.state_mutators.weighted_sample_mutator import WeightedSampleMutator
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator

class RandomPhysicsMutator(StateMutator[GameState]):  #taken from rlgym tools, slightly modified
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding = 100  # Ball radius and car hitbox with biggest radius are both below this
        goal_line_y = 5120
        min_goal_dist = 2000
        i = 0

        for po in [state.ball] + [car.physics for car in state.cars.values()]:
            while True:
                if i == 0:
                    max_z = CEILING_Z - padding
                else:
                    # Cars spawn at max 1/6 ceiling height, because falling from the sky is pointless
                    max_z = (CEILING_Z / 6) - padding

                new_pos = np.random.uniform(
                    [-SIDE_WALL_X + padding, -BACK_WALL_Y + padding, 0 + padding],
                    [SIDE_WALL_X - padding, BACK_WALL_Y - padding, max_z]
                )

               #Make sure ball spawns at least 2000 uu from both goal lines
                if i == 0 and (abs(new_pos[1]) > goal_line_y - min_goal_dist):
                    continue

                # Field edge checks
                if abs(new_pos[0]) + abs(new_pos[1]) >= 8064 - padding:
                    continue

                close_to_wall = (
                    abs(new_pos[0]) >= SIDE_WALL_X - RAMP_HEIGHT or
                    abs(new_pos[1]) >= BACK_WALL_Y - RAMP_HEIGHT or
                    abs(new_pos[0]) + abs(new_pos[1]) >= 8064 - RAMP_HEIGHT
                )
                close_to_floor_or_ceiling = (
                    new_pos[2] <= RAMP_HEIGHT or
                    new_pos[2] >= CEILING_Z - RAMP_HEIGHT
                )

                if close_to_wall and close_to_floor_or_ceiling:
                    continue

                break

            # Assign position and random motion
            po.position = new_pos
            po.linear_velocity = rand_vec3(2300)
            po.angular_velocity = rand_vec3(5)

            # Set rotation matrix for cars only
            if i > 0:
                fw = rand_uvec3()
                up = rand_uvec3()
                right = normalize(np.cross(up, fw))
                up = normalize(np.cross(fw, right))
                rot_mat = np.stack([fw, right, up])
                po.rotation_mtx = rot_mat

            i += 1


    
class RandomStateMutator(StateMutator[GameState]):
    def __init__(self):
        self.mutator = WeightedSampleMutator.from_zipped(
            (KickoffMutator(), 0.6),  #this means that 60% of the time, the ball and the cars will be in kickoff positions
            (RandomPhysicsMutator(), 0.4)   #this means that 40% of the time, the ball and the cars will be in random positions         
        )

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        self.mutator.apply(state, shared_info)