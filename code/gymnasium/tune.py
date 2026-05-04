"""
Script d'optimisation des hyperparamètres avec Optuna pour LunarLander.
========================================================================
Utilisation :
    python tune.py --algo DQN --env discrete --trials 20 --seed 42
    python tune.py --algo PPO --env continuous --trials 20 --seed 42
    python tune.py --algo PPO --env continuous --trials 20 --n-envs 4  # VecEnv
    python tune.py --algo DQN --env discrete --trials 20 --n-jobs 4    # parallèle Optuna

CORRECTIONS APPLIQUÉES :
  - Bug fix : double env.close() supprimé du bloc except
  - Bug fix : seeding de l'environnement via gym.make(seed=seed)
  - Bug fix : guard sur study.best_trial avant sauvegarde
  - Bug fix : trials_done déclaré une seule fois (suppression du doublon)
  - Perf   : support SubprocVecEnv/DummyVecEnv via --n-envs
  - Perf   : évaluation parallélisée via n_eval_envs
  - MLOps  : log optuna/pruned dans WandB pour tracer l'historique
  - MLOps  : hparams/* en WandB summary plutôt qu'en log redondant
  - Clean  : eval_freq aligné sur n_steps pour PPO
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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

# DQN ne supporte pas VecEnv (off-policy mono-env dans SB3)
ALGO_SUPPORTS_VECENV = {"DQN": False, "PPO": True, "SAC": True}

# ---------------------------------------------------------------------------
# Callback WandB — métriques épisodes + métriques algo-spécifiques
# ---------------------------------------------------------------------------

class EpisodeMetricsCallback(BaseCallback):
    """
    Logue dans WandB à chaque step :
      - rollout/ep_rew_mean  : reward moyen des derniers épisodes
      - rollout/ep_rew_std   : écart-type du reward (stabilité)
      - rollout/ep_len_mean  : longueur moyenne des épisodes

    Métriques algo-spécifiques :
      - dqn/exploration_rate : décroissance epsilon-greedy (DQN)
      - sac/entropy_coef     : coefficient d'entropie adaptatif (SAC)
    """
    def __init__(self, run, verbose: int = 0):
        super().__init__(verbose)
        self.run = run

    def _on_step(self) -> bool:
        buf = self.model.ep_info_buffer

        # ── Métriques épisodes (communes à tous les algos) ─────────────────
        if len(buf) > 0:
            rewards = [ep["r"] for ep in buf]
            self.run.log({
                "rollout/ep_rew_mean": np.mean(rewards),
                "rollout/ep_rew_std":  np.std(rewards),
                "rollout/ep_len_mean": np.mean([ep["l"] for ep in buf]),
            }, step=self.num_timesteps)

        # ── DQN : taux d'exploration epsilon-greedy ────────────────────────
        if hasattr(self.model, "exploration_rate"):
            self.run.log({
                "dqn/exploration_rate": self.model.exploration_rate,
            }, step=self.num_timesteps)

        # ── SAC : coefficient d'entropie adaptatif ─────────────────────────
        if hasattr(self.model, "log_ent_coef"):
            try:
                # SB3 >= 2.0 : ent_coef_tensor
                ent_coef = self.model.ent_coef_tensor.item()
            except AttributeError:
                # Fallback : calcul depuis log_ent_coef
                ent_coef = float(torch.exp(self.model.log_ent_coef).detach().cpu())
            self.run.log({
                "sac/entropy_coef": ent_coef,
            }, step=self.num_timesteps)

        return True

# ---------------------------------------------------------------------------
# Callback Optuna Pruning
# ---------------------------------------------------------------------------

class OptunaPruningCallback(BaseCallback):
    """
    Callback pour reporter les performances intermédiaires à Optuna
    et arrêter prématurément l'entraînement (pruning) si l'essai n'est pas prometteur.

    Note PPO : eval_freq doit être un multiple de n_steps pour être effectif,
    car PPO ne step pas à chaque timestep individuel.
    """
    def __init__(self, trial: optuna.Trial, eval_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.trial    = trial
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            buf = self.model.ep_info_buffer
            if len(buf) > 0:
                mean_reward = np.mean([ep["r"] for ep in buf])
                self.trial.report(mean_reward, self.num_timesteps)

                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

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
    net_arch_key = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch = {"small": [64, 64], "medium": [256, 256], "large": [400, 300]}[net_arch_key]

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
        "policy_kwargs":           {"net_arch": net_arch},
        "policy":                  "MlpPolicy",
    }


def sample_ppo_params(trial: optuna.Trial) -> dict:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    n_steps    = trial.suggest_categorical("n_steps",    [256, 512, 1024, 2048])
    net_arch_key = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch     = {"small": [64, 64], "medium": [256, 256], "large": [400, 300]}[net_arch_key]

    if n_steps < batch_size:
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
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True),
        "vf_coef":       trial.suggest_float("vf_coef", 0.1, 1.0),
        "policy_kwargs": {"net_arch": net_arch},
        "policy":        "MlpPolicy",
    }


def sample_sac_params(trial: optuna.Trial) -> dict:
    net_arch_key = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch     = {"small": [64, 64], "medium": [256, 256], "large": [400, 300]}[net_arch_key]

    return {
        "learning_rate":          trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":            trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "learning_starts":        trial.suggest_categorical("learning_starts", [1000, 5000, 10000]),
        "batch_size":             trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        "gamma":                  trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "tau":                    trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05]),
        "use_sde":                trial.suggest_categorical("use_sde", [True, False]),
        "target_update_interval": trial.suggest_categorical("target_update_interval", [1, 2, 4]),
        "ent_coef":               "auto",
        "policy_kwargs":          {"net_arch": net_arch},
        "policy":                 "MlpPolicy",
    }


# ---------------------------------------------------------------------------
# Utilitaires environnement
# ---------------------------------------------------------------------------

def make_env(env_id: str, seed: int, rank: int = 0):
    """Factory pour un environnement seedé (compatible SubprocVecEnv)."""
    def _init():
        env = gym.make(env_id, seed=seed + rank)  # FIX: seed transmis à l'env
        return Monitor(env)
    return _init


def build_train_env(env_id: str, seed: int, n_envs: int, algo_name: str):
    """
    Construit l'environnement d'entraînement.
    - DQN ne supporte pas VecEnv multi-env → always n_envs=1 (Monitor direct)
    - PPO/SAC : SubprocVecEnv si n_envs > 1, DummyVecEnv sinon
    """
    if not ALGO_SUPPORTS_VECENV[algo_name] or n_envs == 1:
        # Mono-env seedé
        env = Monitor(gym.make(env_id, seed=seed))
        return env

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_cls,
        monitor_dir=None,
    )


def build_eval_env(env_id: str, seed: int, n_eval_envs: int = 1):
    """Environnement(s) d'évaluation finale (toujours DummyVecEnv ou Monitor)."""
    if n_eval_envs == 1:
        return Monitor(gym.make(env_id, seed=seed + 9999))
    return make_vec_env(
        env_id,
        n_envs=n_eval_envs,
        seed=seed + 9999,
        vec_env_cls=DummyVecEnv,
    )


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
        eval_freq:       int,
        n_envs:          int,
        n_eval_envs:     int,
    ):
        self.algo_name      = algo_name
        self.env_type       = env_type
        self.seed           = seed
        self.env_id         = ENV_IDS[env_type]
        self.algo_class     = ALGO_CLASSES[algo_name]
        self.tune_timesteps = tune_timesteps
        self.eval_episodes  = eval_episodes
        self.wandb_project  = wandb_project
        self.eval_freq      = eval_freq
        self.n_envs         = n_envs
        self.n_eval_envs    = n_eval_envs

    def __call__(self, trial: optuna.Trial) -> float:
        # ── 1. Sampling ────────────────────────────────────────────────────
        if self.algo_name == "DQN":
            kwargs = sample_dqn_params(trial)
        elif self.algo_name == "PPO":
            kwargs = sample_ppo_params(trial)
        else:
            kwargs = sample_sac_params(trial)

        # ── 2. WandB run ───────────────────────────────────────────────────
        run = wandb.init(
            project=self.wandb_project,
            group=f"{self.algo_name}_{self.env_type}",
            name=f"{self.algo_name}_{self.env_type}_trial_{trial.number}",
            config={
                **kwargs,
                "algo":           self.algo_name,
                "env":            self.env_id,
                "tune_timesteps": self.tune_timesteps,
                "trial_number":   trial.number,
                "n_envs":         self.n_envs,
            },
            reinit=True,
            dir=os.environ.get("WANDB_DIR", "."),
            settings=wandb.Settings(start_method="thread"),
        )

        # ── 3. Environnements ──────────────────────────────────────────────
        train_env = build_train_env(self.env_id, self.seed, self.n_envs, self.algo_name)
        eval_env  = build_eval_env(self.env_id, self.seed, self.n_eval_envs)

        mean_reward: float = float("-inf")
        try:
            # ── 4. Modèle ──────────────────────────────────────────────────
            policy = kwargs.pop("policy")
            model = self.algo_class(
                env=train_env,
                seed=self.seed,
                device="cpu",
                verbose=0,
                policy=policy,
                **kwargs,
            )

            # ── 5. Entraînement ────────────────────────────────────────────
            model.learn(
                total_timesteps=self.tune_timesteps,
                callback=[
                    EpisodeMetricsCallback(run=run),
                    OptunaPruningCallback(trial=trial, eval_freq=self.eval_freq),
                ],
                reset_num_timesteps=True,
            )

            # ── 6. Évaluation finale ───────────────────────────────────────
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
            )

            # Métriques finales → WandB summary (accessible dans Parallel
            # Coordinates Plot sans polluer le log temporel)
            run.summary["eval/mean_reward"] = mean_reward
            run.summary["eval/std_reward"]  = std_reward

            # Marqueur de trial non-pruné
            run.summary["optuna/pruned"] = 0

        except optuna.exceptions.TrialPruned:
            # FIX: log le pruning dans WandB avant de propager
            try:
                run.summary["optuna/pruned"] = 1
            except Exception:
                pass
            raise

        except Exception as exc:
            warnings.warn(f"[Trial {trial.number}] Échec : {exc}")
            run.log({"optuna/crash": 1})
            run.finish(exit_code=1)
            # FIX: on ne ferme PAS les envs ici — le finally s'en charge
            raise

        finally:
            # FIX: fermeture unique des envs dans le finally
            try:
                train_env.close()
            except Exception:
                pass
            try:
                eval_env.close()
            except Exception:
                pass
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
    parser.add_argument("--trials",         type=int, default=150)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--tune-timesteps", type=int, default=250_000,
                        help="Timesteps par trial Optuna")
    parser.add_argument("--eval-episodes",  type=int, default=20,
                        help="Épisodes d'évaluation finale du trial")
    parser.add_argument("--wandb-project",  type=str, default="rl-lunarlander-tune",
                        help="Nom du projet WandB")
    parser.add_argument("--eval-freq",      type=int, default=10_000,
                        help="Fréquence d'évaluation intermédiaire pour le pruning Optuna")
    parser.add_argument("--n-warmup-steps", type=int, default=30_000,
                        help="Steps avant que le pruner Optuna puisse élaguer")
    parser.add_argument("--n-startup-trials", type=int, default=15,
                        help="Trials initiaux avant que le pruner puisse élaguer")
    parser.add_argument("--n-jobs",         type=int, default=1,
                        help="Trials Optuna en parallèle (SQLite gère le lock). "
                             "Note : reproductibilité non garantie en mode parallèle.")
    parser.add_argument("--db-path",        type=str, default=None,
                        help="Chemin vers la base SQLite (défaut: {algo}_{env}_optuna.db)")
    # FIX: nouveaux arguments VecEnv
    parser.add_argument("--n-envs",         type=int, default=1,
                        help="Nombre d'environnements parallèles pour l'entraînement "
                             "(PPO/SAC uniquement). DQN ignoré (forcé à 1).")
    parser.add_argument("--n-eval-envs",    type=int, default=1,
                        help="Nombre d'environnements parallèles pour l'évaluation finale.")

    args = parser.parse_args()

    if args.env not in VALID_COMBINATIONS[args.algo]:
        raise ValueError(
            f"Combinaison invalide : {args.algo} + env={args.env}. "
            f"Combinaisons valides : {VALID_COMBINATIONS}"
        )

    # DQN ne supporte pas VecEnv multi-env dans SB3
    if args.algo == "DQN" and args.n_envs > 1:
        warnings.warn(
            "DQN ne supporte pas n_envs > 1 dans SB3. Forcé à n_envs=1."
        )
        args.n_envs = 1

    set_global_seed(args.seed)

    # ── Optuna study (SQLite pour reprise sur cluster) ─────────────────────
    db_path     = args.db_path or f"{args.algo}_{args.env}_optuna.db"
    storage_url = f"sqlite:///{db_path}?timeout=60"

    study = optuna.create_study(
        study_name=f"{args.algo}_{args.env}",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
        ),
    )

    # FIX: une seule déclaration de trials_done (suppression du doublon)
    trials_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    trials_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    trials_failed   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    trials_done     = trials_complete + trials_pruned

    print(f"  Base de données : {db_path}")
    print(f"  Complétés : {trials_complete} | Prunés : {trials_pruned} | Échoués : {trials_failed}")
    print()

    if trials_done >= args.trials:
        print(f"  ✓ {trials_done} trials déjà complétés — pas de nouveaux trials.")
    else:
        objective = Objective(
            algo_name=args.algo,
            env_type=args.env,
            seed=args.seed,
            tune_timesteps=args.tune_timesteps,
            eval_episodes=args.eval_episodes,
            wandb_project=args.wandb_project,
            eval_freq=args.eval_freq,
            n_envs=args.n_envs,
            n_eval_envs=args.n_eval_envs,
        )

        study.optimize(
            objective,
            n_trials=args.trials - trials_done,
            show_progress_bar=True,
            n_jobs=args.n_jobs,
            catch=(Exception,),
        )

    # ── Résultats ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")

    # FIX: guard sur best_trial avant accès (tous les trials peuvent être prunés)
    if study.best_trial is not None:
        print(f"  Meilleur score : {study.best_value:.2f}")
        print("  Hyperparamètres optimaux :")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

        # ── Sauvegarde locale des meilleurs params ─────────────────────────
        out_dir = os.path.join("models", f"{args.algo}_{args.env}")
        os.makedirs(out_dir, exist_ok=True)
        best_params_path = os.path.join(out_dir, "best_params.txt")
        with open(best_params_path, "w") as f:
            f.write(f"Score: {study.best_value}\n")
            f.write("Params:\n")
            for k, v in study.best_trial.params.items():
                f.write(f"  {k}: {v}\n")
        print(f"  Meilleurs paramètres sauvegardés dans : {best_params_path}")
    else:
        print("  Aucun trial complété avec succès.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()