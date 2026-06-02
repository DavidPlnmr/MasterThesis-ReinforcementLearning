import numpy as np
from rlgym_ppo.util import MetricsLogger

g_combined_reward = None  # type: LogCombinedReward

class RewardMetricsLogger(MetricsLogger):

    def _collect_metrics(self, game_state) -> list:
        global g_combined_reward
        prev_rewards = []
        if g_combined_reward is not None and g_combined_reward.prev_rewards is not None:
            prev_rewards = g_combined_reward.prev_rewards.copy()

        # Récupère la vélocité du premier agent disponible
        first_agent = list(game_state.cars.keys())[0]
        car = game_state.cars[first_agent]

        return [
            car.physics.linear_velocity,
            prev_rewards
        ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps) -> None:
        global g_combined_reward

        avg_linvel = np.zeros(3)
        n_rewards = len(g_combined_reward.reward_functions) if g_combined_reward else 0
        avg_rewards = np.zeros(n_rewards)
        valid_reward_count = 0

        for metric_array in collected_metrics:
            avg_linvel += metric_array[0]

            prev_rewards = metric_array[1]
            if len(prev_rewards) == n_rewards:
                avg_rewards += prev_rewards
                valid_reward_count += 1

        avg_linvel /= len(collected_metrics)
        if valid_reward_count > 0:
            avg_rewards /= valid_reward_count

        num_days_played = cumulative_timesteps / (120 / 8) / 60 / 60 / 24

        report = {
            "x_vel": avg_linvel[0],
            "y_vel": avg_linvel[1],
            "z_vel": avg_linvel[2],
            "Cumulative Timesteps": cumulative_timesteps,
            "Days": num_days_played,
            "Years played": num_days_played / 365,
        }

        # Ajoute chaque reward individuellement
        if g_combined_reward is not None:
            for i, func in enumerate(g_combined_reward.reward_functions):
                report[f"rewards/{func.__class__.__name__}"] = avg_rewards[i]

        wandb_run.log(report)