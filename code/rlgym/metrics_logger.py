# metrics_logger.py
import os
import numpy as np

ALGO = os.environ.get("RL_ALGO", "ppo")  # "ppo" ou "sac"

if ALGO == "sac":

    from rlgym_sac.util import MetricsLogger
else:
    from rlgym_ppo.util import MetricsLogger


g_combined_reward = None  # type: LogCombinedReward


class RewardMetricsLogger(MetricsLogger):
    def _collect_metrics(self, game_state, done=None) -> list:
        # snapshot des contributions courantes ; .copy() est crucial
        # car prev_rewards est réécrit à chaque step
        return [g_combined_reward.prev_rewards.copy()]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps) -> None:
        global g_combined_reward
        avg_rewards = np.zeros(len(g_combined_reward.reward_functions))
        for metric_array in collected_metrics:
            avg_rewards += metric_array[0]
        avg_rewards /= len(collected_metrics)
        num_days_played = cumulative_timesteps / (120 / 8) / 60 / 60 / 24
        report = {"Days": num_days_played,
                  "Years": num_days_played / 365}
        for i in range(len(g_combined_reward.reward_functions)):
            report["reward/" + g_combined_reward.reward_names[i]] = avg_rewards[i]
        if wandb_run is not None:
            wandb_run.log(report)