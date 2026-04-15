"""
Script d'optimisation des hyperparamètres avec Optuna pour LunarLander.
========================================================================
Utilisation :
    python tune.py --algo DQN --env discrete --trials 20 --seed 42
    python tune.py --algo PPO --env continuous --trials 20 --seed 42
"""

import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import optuna
import wandb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# ---------------------------------------------------------------------------
# Configuration centrale
# ---------------------------------------------------------------------------

ENV_IDS = {
    "discrete":   "LunarLander-v3",
    "continuous": "LunarLanderContinuous-v3",
}

ALGO_CLASSES = {"DQN": DQN, "PPO": PPO, "SAC": SAC}

VALID_COMBINATIONS = {
    "DQN": ["discrete"],
    "PPO": ["discrete", "continuous"],
    "SAC": ["continuous"],
}

# ---------------------------------------------------------------------------
# Seeding global
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------------------------------------------------------------------
# Optuna Objective
# ---------------------------------------------------------------------------

def sample_dqn_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [0, 1000, 5000]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
        "target_update_interval": trial.suggest_categorical("target_update_interval", [100, 250, 500, 1000]),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        "policy": "MlpPolicy",
    }

def sample_ppo_params(trial: optuna.Trial) -> dict:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    if n_steps < batch_size: n_steps = batch_size
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_categorical("n_epochs", [1, 3, 5, 10, 20]),
        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
        "ent_coef": trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4]),
        "policy": "MlpPolicy",
    }

def sample_sac_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [1000, 5000, 100000]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "tau": trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05]),
        "ent_coef": "auto",
        "policy": "MlpPolicy",
    }

class Objective:
    def __init__(self, algo_name: str, env_type: str, seed: int, tune_timesteps: int, eval_episodes: int):
        self.algo_name = algo_name
        self.env_type = env_type
        self.seed = seed
        self.env_id = ENV_IDS[env_type]
        self.algo_class = ALGO_CLASSES[algo_name]
        self.tune_timesteps = tune_timesteps
        self.eval_episodes = eval_episodes

    def __call__(self, trial: optuna.Trial) -> float:
        if self.algo_name == "DQN":
            kwargs = sample_dqn_params(trial)
        elif self.algo_name == "PPO":
            kwargs = sample_ppo_params(trial)
        elif self.algo_name == "SAC":
            kwargs = sample_sac_params(trial)

        # 1. Initialiser une run WandB spécifique pour ce trial
        run = wandb.init(
            project="rl-lunarlander-tune",
            group=f"{self.algo_name}_{self.env_type}",
            name=f"trial_{trial.number}",
            config=kwargs,
            sync_tensorboard=True, # Essentiel pour extraire les courbes de SB3
            reinit=True
        )

        # 2. Créer l'environnement
        env = gym.make(self.env_id)
        env = Monitor(env)

        try:
            # 3. Initialiser le modèle avec le paramètre tensorboard_log
            model = self.algo_class(
                env=env, 
                seed=self.seed, 
                tensorboard_log=f"runs/{run.id}", 
                **kwargs
            )
            
            # 4. Apprendre en passant le callback de WandB
            model.learn(
                total_timesteps=self.tune_timesteps,
                callback=WandbCallback(gradient_save_freq=0, verbose=0)
            )
            
            # 5. Évaluation
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=self.eval_episodes)
            wandb.log({"optuna/mean_reward": mean_reward})
            
        except Exception as e:
            print(f"Échec de l'entraînement de ce trial à cause de: {e}")
            env.close()
            run.finish()
            raise optuna.exceptions.TrialPruned()
            
        finally:
            env.close()
            run.finish()

        return mean_reward

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "PPO", "SAC"])
    parser.add_argument("--env", type=str, required=True, choices=["discrete", "continuous"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune-timesteps", type=int, default=100_000, help="Nombre de timesteps par trial Optuna")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Nombre d'épisodes pour l'évaluation")
    args = parser.parse_args()

    if args.env not in VALID_COMBINATIONS[args.algo]:
        raise ValueError(f"Combinaison invalide : {args.algo} avec l'environnement {args.env}.")

    set_global_seed(args.seed)

    print(f"--- Début du Tuning Optuna ---")
    print(f"Algo : {args.algo} | Env : {args.env} | Trials : {args.trials}")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé par PyTorch : {device}")
    if torch.cuda.is_available():
        print(f"Nom du GPU : {torch.cuda.get_device_name(device)}")
        
    # 1. Définir le chemin de la base de données locale (SQLite)
    db_name = f"{args.algo}_{args.env}_optuna.db"
    storage_url = f"sqlite:///{db_name}"
    
    # 2. Assigner le storage et activer 'load_if_exists=True'
    study = optuna.create_study(
        study_name=f"{args.algo}_{args.env}",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner()
    )

    # 3. Afficher le nombre d'essais déjà complétés
    trials_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"-> Reprise de l'étude (Base de données : {db_name})")
    print(f"-> Essais déjà complétés : {trials_done}")

    objective = Objective(args.algo, args.env, args.seed, args.tune_timesteps, args.eval_episodes)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\n=> Meilleur run: Score de {study.best_value}")
    print("Hyperparamètres optimaux :")
    for key, val in study.best_trial.params.items():
        print(f"  {key}: {val}")

    # Sauvegarde
    out_dir = os.path.join("models", f"{args.algo}_{args.env}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best_params.txt"), "w") as f:
        f.write(f"Score: {study.best_value}\n")
        f.write("Params:\n")
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()
