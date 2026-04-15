"""
Script d'optimisation des hyperparamètres avec Optuna pour LunarLander.
========================================================================
Utilisation :
    python tune.py --algo DQN --env discrete --trials 20 --seed 42
    python tune.py --algo PPO --env continuous --trials 20 --seed 42

Fixes appliqués :
    - wandb.tensorboard.patch()  → appelé UNE SEULE FOIS avant l'étude,
      plus dans l'objective (évite les re-patches qui perdent des données).
    - Crashs traités comme PRUNED → désormais levés en tant que vraies
      exceptions pour ne pas polluer l'étude Optuna.
    - Environnement seedé + seeding CUDA complet.
    - tb_log_name fixé pour un chemin TensorBoard stable (pas de timestamp).
    - Contrainte PPO n_steps >= batch_size gérée proprement.
    - device="auto" : GPU utilisé si disponible (nécessite .sif buildé
      depuis pytorch:2.1.2-cuda11.8 pour compatibilité sm_61 / TITAN Xp).
"""

import argparse
import os
import random
import warnings

import gymnasium as gym
import numpy as np
import optuna
import torch
import wandb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
# Configuration centrale
# ---------------------------------------------------------------------------

ENV_IDS = {
    "discrete":   "LunarLander-v2",
    "continuous": "LunarLanderContinuous-v2",
}

ALGO_CLASSES = {"DQN": DQN, "PPO": PPO, "SAC": SAC}

VALID_COMBINATIONS = {
    "DQN": ["discrete"],
    "PPO": ["discrete", "continuous"],
    "SAC": ["continuous"],
}

class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, run, verbose: int = 0):
        super().__init__(verbose)
        self.run = run

    def _on_step(self) -> bool:
        buf = self.model.ep_info_buffer
        if len(buf) > 0:
            self.run.log({
                "rollout/ep_rew_mean": np.mean([ep["r"] for ep in buf]),
                "rollout/ep_len_mean": np.mean([ep["l"] for ep in buf]),
            }, step=self.num_timesteps)
        return True

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
# Sampling des hyperparamètres
# ---------------------------------------------------------------------------

def sample_dqn_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":           trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":             trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "learning_starts":         trial.suggest_categorical("learning_starts", [0, 1000, 5000]),
        "batch_size":              trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma":                   trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "train_freq":              trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
        "target_update_interval":  trial.suggest_categorical("target_update_interval", [100, 250, 500, 1000]),
        "exploration_fraction":    trial.suggest_float("exploration_fraction", 0.05, 0.5),
        "exploration_final_eps":   trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        "policy": "MlpPolicy",
    }


def sample_ppo_params(trial: optuna.Trial) -> dict:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    n_steps    = trial.suggest_categorical("n_steps",    [256, 512, 1024, 2048])

    # Contrainte dure : n_steps doit être >= batch_size pour que PPO puisse
    # construire au moins un mini-batch complet.
    if n_steps < batch_size:
        # On signale le trial comme invalide proprement ; Optuna ne le compte
        # pas comme un vrai essai et en relance un autre.
        raise optuna.exceptions.TrialPruned(
            f"n_steps ({n_steps}) < batch_size ({batch_size}) : config invalide."
        )

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps":       n_steps,
        "batch_size":    batch_size,
        "n_epochs":      trial.suggest_categorical("n_epochs", [1, 3, 5, 10, 20]),
        "gamma":         trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "gae_lambda":    trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "clip_range":    trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4]),
        "policy": "MlpPolicy",
    }


def sample_sac_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":   trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":     trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [1000, 5000, 10000]),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        "gamma":           trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "tau":             trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05]),
        "ent_coef":        "auto",
        "policy": "MlpPolicy",
    }


# ---------------------------------------------------------------------------
# Objective Optuna
# ---------------------------------------------------------------------------

class Objective:
    def __init__(
        self,
        algo_name:       str,
        env_type:        str,
        seed:            int,
        tune_timesteps:  int,
        eval_episodes:   int,
        wandb_project:   str,
    ):
        self.algo_name      = algo_name
        self.env_type       = env_type
        self.seed           = seed
        self.env_id         = ENV_IDS[env_type]
        self.algo_class     = ALGO_CLASSES[algo_name]
        self.tune_timesteps = tune_timesteps
        self.eval_episodes  = eval_episodes
        self.wandb_project  = wandb_project

    def __call__(self, trial: optuna.Trial) -> float:
        # ── 1. Sampling ────────────────────────────────────────────────────
        if self.algo_name == "DQN":
            kwargs = sample_dqn_params(trial)
        elif self.algo_name == "PPO":
            kwargs = sample_ppo_params(trial)  # peut lever TrialPruned
        else:
            kwargs = sample_sac_params(trial)


        # ── 3. WandB run ───────────────────────────────────────────────────
        # sync_tensorboard=True + le patch fait UNE SEULE FOIS avant l'étude
        # (voir main()) suffit ; pas besoin de re-patcher ici.
        run = wandb.init(
            project=self.wandb_project,
            group=f"{self.algo_name}_{self.env_type}",
            name=f"trial_{trial.number}",
            config={
                **kwargs,
                "algo":           self.algo_name,
                "env":            self.env_id,
                "tune_timesteps": self.tune_timesteps,
                "trial_number":   trial.number,
            },
            reinit=True,
            # Évite que WandB crée un sous-dossier wandb/ dans le répertoire
            # courant de chaque trial — utile sur les clusters.
            dir=os.environ.get("WANDB_DIR", "."),
        )

        # ── 4. Environnement ───────────────────────────────────────────────
        env = Monitor(gym.make(self.env_id))

        mean_reward: float = float("-inf")
        try:
            # ── 5. Modèle ──────────────────────────────────────────────────
            # device="auto" : SB3 utilise le GPU si disponible (CUDA 11.8 +
            # sm_61 supporté avec pytorch:2.1.2-cuda11.8 dans le .sif).
            model = self.algo_class(
                env=env,
                seed=self.seed,
                device="auto",
                verbose=0,
                **{k: v for k, v in kwargs.items() if k != "policy"},
                policy=kwargs["policy"],
            )

            # ── 6. Entraînement ────────────────────────────────────────────
            model.learn(
                total_timesteps=self.tune_timesteps,
                callback=EpisodeMetricsCallback(run),
                reset_num_timesteps=True,
            )

            # ── 7. Évaluation ──────────────────────────────────────────────
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=self.eval_episodes, deterministic=True
            )
            run.log({
                "optuna/mean_reward": mean_reward,
                "optuna/std_reward":  std_reward,
                "optuna/trial":       trial.number,
            })

        except optuna.exceptions.TrialPruned:
            # Re-propagation propre (e.g. contrainte PPO n_steps/batch_size)
            raise

        except Exception as exc:
            # Vrai crash (CUDA, OOM, etc.) : on le logue et on le remonte
            # comme un échec, PAS comme un pruning, pour ne pas biaiser TPE.
            warnings.warn(f"[Trial {trial.number}] Échec : {exc}")
            run.log({"optuna/crash": 1})
            run.finish(exit_code=1)
            env.close()
            raise  # ← remonte l'exception réelle → Optuna marque FAIL

        finally:
            env.close()
            # run.finish() est idempotent ; on l'appelle dans tous les cas
            try:
                run.finish()
            except Exception:
                pass

        return mean_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",           type=str, required=True, choices=["DQN", "PPO", "SAC"])
    parser.add_argument("--env",            type=str, required=True, choices=["discrete", "continuous"])
    parser.add_argument("--trials",         type=int, default=20)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--tune-timesteps", type=int, default=100_000,
                        help="Timesteps par trial Optuna")
    parser.add_argument("--eval-episodes",  type=int, default=5,
                        help="Épisodes d'évaluation finale du trial")
    parser.add_argument("--wandb-project",  type=str, default="rl-lunarlander-tune",
                        help="Nom du projet WandB")
    args = parser.parse_args()

    if args.env not in VALID_COMBINATIONS[args.algo]:
        raise ValueError(
            f"Combinaison invalide : {args.algo} + env={args.env}. "
            f"Combinaisons valides : {VALID_COMBINATIONS}"
        )

    set_global_seed(args.seed)

    # ── Device info ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Tuning Optuna : {args.algo} | {args.env} | {args.trials} trials")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"  GPU détecté : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA {torch.version.cuda} — sm_61 : {'sm_61' in torch.cuda.get_arch_list()}")
    else:
        print("  CPU uniquement.")
    print()

    # ── WandB tensorboard patch — UNE SEULE FOIS avant l'étude ────────────
    # Pointer sur le dossier racine commun à tous les trials de cette étude.
    root_tb_dir = f"runs/{args.algo}_{args.env}"
    os.makedirs(root_tb_dir, exist_ok=True)
    wandb.tensorboard.patch(root_logdir=root_tb_dir)

    # ── Optuna study (SQLite pour reprise sur cluster) ────────────────────
    db_path     = f"{args.algo}_{args.env}_optuna.db"
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"{args.algo}_{args.env}",
        storage=storage_url,
        load_if_exists=True,          # reprise automatique si le job SLURM redémarre
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20_000),
    )

    trials_done = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    print(f"  Base de données : {db_path}")
    print(f"  Essais déjà complétés : {trials_done}")
    print()

    objective = Objective(
        algo_name=args.algo,
        env_type=args.env,
        seed=args.seed,
        tune_timesteps=args.tune_timesteps,
        eval_episodes=args.eval_episodes,
        wandb_project=args.wandb_project,
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=True,
        # Continuer même si un trial lève une exception non-Optuna
        catch=(Exception,),
    )

    # ── Résultats ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if study.best_trial is not None:
        print(f"  Meilleur score : {study.best_value:.2f}")
        print("  Hyperparamètres optimaux :")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")
    else:
        print("  Aucun trial complété avec succès.")
    print(f"{'='*60}\n")

    # ── Sauvegarde locale des meilleurs params ─────────────────────────────
    out_dir = os.path.join("models", f"{args.algo}_{args.env}")
    os.makedirs(out_dir, exist_ok=True)
    best_params_path = os.path.join(out_dir, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(f"Score: {study.best_value}\n")
        f.write("Params:\n")
        for k, v in study.best_trial.params.items():
            f.write(f"  {k}: {v}\n")
    print(f"  Meilleurs paramètres sauvegardés dans : {best_params_path}")


if __name__ == "__main__":
    main()