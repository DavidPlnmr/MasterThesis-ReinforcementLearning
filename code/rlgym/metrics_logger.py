# metrics_logger.py
import numpy as np
from rlgym_ppo.util import MetricsLogger

g_combined_reward = None  # type: LogCombinedReward


class RewardMetricsLogger(MetricsLogger):
    def _collect_metrics(self, game_state) -> list:
        # snapshot des contributions courantes ; .copy() est crucial
        # car prev_rewards est réécrit à chaque step
        return [g_combined_reward.prev_rewards.copy()]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps) -> None:
        global g_combined_reward
        avg_rewards = np.zeros(len(g_combined_reward.reward_functions))
        for metric_array in collected_metrics:
            avg_rewards += metric_array[0]
        avg_rewards /= len(collected_metrics)

        days = cumulative_timesteps / (120 * 60 * 60)  # Convert timesteps to in-game days (assuming 120 steps/s)

        report = {"Days": days,
                  "Years": days / 365}  # Convert timesteps to in-game days (assuming 120 steps/s)
        for i in range(len(g_combined_reward.reward_functions)):
            name = g_combined_reward.reward_functions[i].__class__.__name__
            report["reward/" + name] = avg_rewards[i]

        if wandb_run is not None:
            wandb_run.log(report)