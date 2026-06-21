import argparse
import os


RUN_ON_SLURM = True # If I run this on Slurm, I set this to True to avoid issues with KBHit and GPU/CPU tensor checkpoints. If I run it locally, I set this to False to disable WandB logging and enable rendering.

if RUN_ON_SLURM:
    import patch_kbhit # Patch pour éviter les problèmes de KBHit sur Slurm qui attendent une entrée clavier. Doit être importé avant rlgym_ppo.
    
    
else :
    import patch_torch_gpu # Patch pour éviter les problèmes de checkpoints de GPU/CPU tensors. Doit être importé avant rlgym_ppo.
    os.environ["WANDB_MODE"] = "disabled"  # Disable WandB logging output

project_name = "rlgym_ppo_example"

# from metrics_logger import RewardMetricsLogger
# import metrics_logger



def build_rlgym_v2_env():
    
    from rewards import CloseRangeFaceBallReward, VelocityBallToGoalReward, LogCombinedReward, ZeroSumReward, KickoffReward, EnergyReward, InAirReward, FaceBallReward
    import metrics_logger
    from statemutator import RandomStateMutator
    


    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper

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
    

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    VelocityBallToGoalReward_ZS = ZeroSumReward(VelocityBallToGoalReward(), team_spirit=0, opp_scale=1)
    BoostKeepReward_ZS = ZeroSumReward(BoostKeepReward(), team_spirit=0, opp_scale=1)
    BoostChangeReward_ZS = ZeroSumReward(BoostChangeReward(gain_weight=1.0, lose_weight=0.0), team_spirit=0, opp_scale=1)

    reward_fn = LogCombinedReward(
        (GoalReward(), 750.0),
        (AdvancedTouchReward(touch_reward=0.5, acceleration_reward=1.0, use_touch_count=True), 7.5),
        (VelocityBallToGoalReward_ZS, 25.0, "VelocityBallToGoalReward_ZS"),
        (InAirReward(), 0.05),
        (KickoffReward(), 25.0),
        (BoostKeepReward_ZS, 2.5, "BoostKeepReward_ZS"),
        (BoostChangeReward_ZS, 2.0, "BoostChangeReward_ZS"),
        (EnergyReward(), 0.5),
        (AerialDistanceReward(), 12.0),
        (WavedashReward(), 10.0),
        (DemoReward(), 10.0)
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

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from metrics_logger import RewardMetricsLogger

    parser = argparse.ArgumentParser(description="Train an RLGym PPO agent.")
    
    parser.add_argument(
        "--n-proc", 
        type=int,
        default=8,
        help="Nombre de processus d'environnement à utiliser pour l'entraînement."
    )
    parser.add_argument(
        "--save-every-ts",
        type=int,
        default=25_000_000,
        help="Sauvegarder le modèle tous les N timesteps."
    )
    parser.add_argument(
        "--timesteps-limit",
        type=int,
        default=12_500_000_000,
        help="Limite de timesteps pour l'entraînement."
    )
    args = parser.parse_args()

    from rlgym_ppo import Learner

    # processes
    n_proc = args.n_proc

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    checkpoint_folder = f"data/checkpoints/{project_name}"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    checkpoint_files = os.listdir(checkpoint_folder)
    valid_checkpoints = [f for f in checkpoint_files if f.isdigit()]
    checkpoint_load_folder = os.path.join(checkpoint_folder, max(valid_checkpoints, key=int)) if valid_checkpoints else None


    print(f"Loading checkpoint: {checkpoint_load_folder}")

    ts_per_it = 200_000
    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=RewardMetricsLogger(), # Use the custom metrics logger
                      ppo_batch_size=ts_per_it,  # batch size - much higher than 300K doesn't seem to help most people
                      ts_per_iteration=ts_per_it,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=ts_per_it * 3,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,  # entropy coefficient - this determines the impact of exploration
                      policy_lr=1e-4,  # policy learning rate
                      critic_lr=1e-4,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      gae_gamma=0.995,  # GAE gamma - discount factor for rewards
                      policy_layer_sizes=[1024, 1024, 512, 512],  # policy network
                      critic_layer_sizes=[1024, 1024, 512, 512],  # critic network making it the same size as the policy 
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=args.save_every_ts,  # save every 1M steps
                      timestep_limit=args.timesteps_limit,  # Train for 1B steps
                      checkpoint_load_folder=checkpoint_load_folder,  # Automatically load the latest checkpoint if it exists
                      checkpoints_save_folder=checkpoint_folder,
                      add_unix_timestamp=False,
                      log_to_wandb=RUN_ON_SLURM, # Set this to True if you want to use Weights & Biases for logging.
                      render=not RUN_ON_SLURM,  # Disable rendering if running on Slurm to avoid issues.
                      render_delay=8/120,
                      load_wandb=False,
                      n_checkpoints_to_keep=10
                      )

    build_rlgym_v2_env()

    learner.learn()