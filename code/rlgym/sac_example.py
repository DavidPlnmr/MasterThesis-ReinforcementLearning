import argparse
import os

os.environ["RL_ALGO"] = "sac"

RUN_ON_SLURM = True  # Si je lance sur Slurm, True pour éviter les soucis KBHit et GPU/CPU tensor checkpoints. En local, False pour désactiver WandB et activer le rendu.

if RUN_ON_SLURM:
    import patch_kbhit  # Patch pour éviter les problèmes de KBHit sur Slurm. Doit être importé avant rlgym_sac.
else:
    import patch_torch_gpu  # Patch pour éviter les problèmes de checkpoints GPU/CPU tensors. Doit être importé avant rlgym_sac.
    os.environ["WANDB_MODE"] = "disabled"

project_name = "rlgym_sac_example"


def build_rlgym_v2_env():

    from rewards import CloseRangeFaceBallReward, VelocityBallToGoalReward, LogCombinedReward, ZeroSumReward, KickoffReward, EnergyReward, InAirReward, FaceBallReward
    import metrics_logger
    from statemutator import RandomStateMutator
    import gymnasium as gym

    import numpy as np
    from rlgym.api import RLGym
    
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league.action_parsers import RepeatAction
    from rlgym.rocket_league import common_values
    from patch_rlgymwrapper import FixedRLGymV2GymWrapper  
    from action_parser import ContinuousAction  # <-- swap: LookupTableAction -> ContinuousAction

    from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer

    from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
    from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
    from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import AerialDistanceReward
    from rlgym_tools.rocket_league.reward_functions.boost_keep_reward import BoostKeepReward
    from rlgym_tools.rocket_league.reward_functions.boost_change_reward import BoostChangeReward
    from rlgym_tools.rocket_league.reward_functions.wavedash_reward import WavedashReward
    from rlgym_tools.rocket_league.reward_functions.demo_reward import DemoReward
    

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = action_parser = RepeatAction(ContinuousAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    VelocityBallToGoalReward_ZS = ZeroSumReward(VelocityBallToGoalReward(), team_spirit=0, opp_scale=1)
    BoostKeepReward_ZS = ZeroSumReward(BoostKeepReward(), team_spirit=0, opp_scale=1)
    BoostChangeReward_ZS = ZeroSumReward(BoostChangeReward(gain_weight=1.0, lose_weight=0.0), team_spirit=0, opp_scale=1)

    factor_dividor = 0.01    
    reward_fn = LogCombinedReward(
        (AdvancedTouchReward(touch_reward=1.0, acceleration_reward=0.0), 5 * factor_dividor),
        (VelocityPlayerToBallReward(), 1 * factor_dividor),
        (InAirReward(), 0.1 * factor_dividor),
        (FaceBallReward(), 0.1 * factor_dividor),
    )

    metrics_logger.g_combined_reward = reward_fn

    obs_builder = DefaultObs(zero_padding=None,
                           pos_coef=np.asarray([1 / common_values.SIDE_WALL_X,
                                              1 / common_values.BACK_NET_Y,
                                              1 / common_values.CEILING_Z]),
                           ang_coef=1 / np.pi,
                           lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                           ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                           boost_coef=1 / 100.0)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        RandomStateMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer()
    )

    # return RLGymV2GymWrapper(rlgym_env)
    return FixedRLGymV2GymWrapper(rlgym_env)

if __name__ == "__main__":
    from metrics_logger import RewardMetricsLogger

    parser = argparse.ArgumentParser(description="Train an RLGym SAC agent.")

    parser.add_argument(
        "--n-proc",
        type=int,
        default=8,
        help="Nombre de processus d'environnement à utiliser pour l'entraînement."
    )
    parser.add_argument(
        "--save-every-ts",
        type=int,
        default=1_000_000,
        help="Sauvegarder le modèle tous les N timesteps."
    )
    parser.add_argument(
        "--timesteps-limit",
        type=int,
        default=100_000_000,
        help="Limite de timesteps pour l'entraînement."
    )
    args = parser.parse_args()

    from rlgym_sac import Learner  # <-- swap: rlgym_ppo -> rlgym_sac

    n_proc = args.n_proc
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    checkpoint_folder = f"data/checkpoints/{project_name}"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_files = os.listdir(checkpoint_folder)
    valid_checkpoints = [f for f in checkpoint_files if f.isdigit()]
    checkpoint_load_folder = os.path.join(checkpoint_folder, max(valid_checkpoints, key=int)) if valid_checkpoints else None

    print(f"Loading checkpoint: {checkpoint_load_folder}")

    ts_per_it = 50_000
    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=RewardMetricsLogger(),
                      ts_per_iteration=ts_per_it,                  # cadence de collecte/logging, plus besoin d'égaler un batch
                      exp_buffer_size=1_000_000,                    # vrai replay buffer SAC (pas 2-3x le batch comme en PPO)
                      sac_batch_size=256,                           # taille standard du minibatch tiré du buffer
                      sac_ent_coef='auto',                          # auto-tuning, pas d'équivalent direct à ppo_ent_coef=0.01
                      sac_learning_rate=2e-4,                       # = vos policy_lr/critic_lr PPO pour comparaison équitable
                      sac_learning_starts=10_000,
                      policy_layer_sizes=[1024, 1024, 512, 512],    # même capacité réseau que le PPO
                      critic_layer_sizes=[1024, 1024, 512, 512],
                      sac_gamma=0.99,                                  # discount factor standard
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=args.save_every_ts,
                      timestep_limit=args.timesteps_limit,
                      checkpoint_load_folder=checkpoint_load_folder,
                      checkpoints_save_folder=checkpoint_folder,
                      add_unix_timestamp=False,
                      log_to_wandb=RUN_ON_SLURM,
                      render=not RUN_ON_SLURM,
                      render_delay=8/120,
                      load_wandb=True,
                      wandb_project_name="rlgym-experiments",
                      n_checkpoints_to_keep=5,
                      use_amp=False
                      )

    build_rlgym_v2_env()

    learner.learn()