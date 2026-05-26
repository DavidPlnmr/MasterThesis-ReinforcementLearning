from typing import Dict, Any
import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values

class RandomStateMutator(StateMutator[GameState]):
    """A StateMutator that sets random positions for cars and the ball."""
    
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        # Define spawn location and orientation

        car_x = np.random.uniform(-common_values.SIDE_WALL_X * 0.75, common_values.SIDE_WALL_X * 0.75)
        car_y = np.random.uniform(-common_values.BACK_NET_Y * 0.75, common_values.BACK_NET_Y * 0.75)
        car_z = np.random.uniform(0, common_values.CEILING_Z * 0.75)
        desired_car_pos = np.array([car_x, car_y, car_z], dtype=np.float32)  # x, y, z
        desired_yaw = np.random.uniform(-np.pi, np.pi)  # yaw angle in radians

        # Iterate over all cars in the game
        for car in state.cars.values():
            if car.is_orange:
                # Orange team positions
                pos = desired_car_pos
                yaw = desired_yaw
            else:
                # Blue team positions (inverted)
                pos = -desired_car_pos
                yaw = -desired_yaw

            # Set car physics state
            car.physics.position = pos
            car.physics.euler_angles = np.array([0, 0, yaw], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            # car.boost = np.random.uniform(0, 100)  # random boost amount

        # Set ball physics state
        ball_x = np.random.uniform(-common_values.SIDE_WALL_X * 0.75, common_values.SIDE_WALL_X * 0.75)
        ball_y = np.random.uniform(-common_values.BACK_NET_Y * 0.75, common_values.BACK_NET_Y * 0.75)
        ball_z = np.random.uniform(0, common_values.CEILING_Z * 0.75)  

        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)

        # Randomize ball velocity as well, but you can set this to zero if you want

        speed = common_values.BALL_MAX_SPEED / 10

        ball_lin_vel_x = np.random.uniform(-speed, speed)
        ball_lin_vel_y = np.random.uniform(-speed, speed)
        ball_lin_vel_z = np.random.uniform(-speed, speed)

        state.ball.linear_velocity = np.array([ball_lin_vel_x, ball_lin_vel_y, ball_lin_vel_z], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)