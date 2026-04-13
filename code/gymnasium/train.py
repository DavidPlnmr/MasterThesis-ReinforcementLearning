"""
Benchmark RL : DQN, PPO, SAC sur LunarLander
=============================================
- DQN  + PPO  -> LunarLander-v3              (espace discret)
- PPO  + SAC  -> LunarLanderContinuous-v3    (espace continu)

Lancement :
    python train.py --algo DQN --env discrete --seed 42
    python train.py --algo PPO --env discrete --seed 42
    python train.py --algo PPO --env continuous --seed 42
    python train.py --algo SAC --env continuous --seed 42

Reprise depuis un checkpoint :
    python train.py --algo DQN --env discrete --seed 42 --resume

Ou via le script run_all.sh pour lancer toutes les combinaisons.
"""

import argparse
import os
import random
import time
import glob

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# Configuration centrale
# ---------------------------------------------------------------------------

ENV_IDS = {
    "discrete":   "LunarLander-v3",
    "continuous": "LunarLanderContinuous-v3",
}

# Budget total de timesteps identique pour tous les algos
TOTAL_TIMESTEPS = 500_000

# Fréquence de sauvegarde des checkpoints (en timesteps)
CHECKPOINT_FREQ = 50_000

# Seeds utilisées pour la reproductibilité
SEEDS = [42, 123, 456, 789, 1337]

# Hyperparamètres — valeurs par défaut SB3, sauf mention contraire
HYPERPARAMS = {
    "DQN": {
        "learning_rate":          1e-4,
        "buffer_size":            50_000,
        "learning_starts":        1_000,
        "batch_size":             64,
        "gamma":                  0.99,
        "train_freq":             4,
        "target_update_interval": 250,
        "exploration_fraction":   0.12,
        "exploration_final_eps":  0.05,
        "policy":                 "MlpPolicy",
    },
    "PPO": {
        "learning_rate": 3e-4,
        "n_steps":       1024,
        "batch_size":    64,
        "n_epochs":      4,
        "gamma":         0.999,
        "gae_lambda":    0.98,
        "ent_coef":      0.01,
        "clip_range":    0.2,
        "policy":        "MlpPolicy",
    },
    "SAC": {
        "learning_rate":   3e-4,
        "buffer_size":     50_000,
        "learning_starts": 1_000,
        "batch_size":      256,
        "gamma":           0.99,
        "tau":             0.005,
        "ent_coef":        "auto",
        "policy":          "MlpPolicy",
    },
}

# Combinaisons valides algo <-> environnement
VALID_COMBINATIONS = {
    "DQN": ["discrete"],
    "PPO": ["discrete", "continuous"],
    "SAC": ["continuous"],
}

ALGO_CLASSES = {"DQN": DQN, "PPO": PPO, "SAC": SAC}

WANDB_PROJECT = "rl-lunarlander-benchmark"


# ---------------------------------------------------------------------------
# Seeding global
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Fixe tous les générateurs aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Gestion des checkpoints
# ---------------------------------------------------------------------------

def get_checkpoint_dir(algo: str, env_type: str) -> str:
    return f"models/{algo}_{env_type}/checkpoints"


def get_checkpoint_prefix(algo: str, seed: int) -> str:
    return f"{algo}_seed_{seed}"


def find_latest_checkpoint(algo: str, env_type: str, seed: int) -> tuple[str | None, int]:
    """
    Cherche le checkpoint le plus récent pour une combinaison algo/env/seed.

    Retourne :
        (chemin_du_checkpoint, timesteps_déjà_effectués)
        ou (None, 0) si aucun checkpoint trouvé.

    Convention de nommage SB3 :
        {prefix}_{timesteps}_steps.zip
    """
    checkpoint_dir = get_checkpoint_dir(algo, env_type)
    prefix         = get_checkpoint_prefix(algo, seed)
    pattern        = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")

    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, 0

    # Extraire le nombre de timesteps depuis le nom de fichier
    # Exemple : "DQN_seed_42_25000_steps.zip" -> 25000
    def extract_timesteps(path: str) -> int:
        basename = os.path.basename(path)           # "DQN_seed_42_25000_steps.zip"
        no_ext   = basename.replace(".zip", "")     # "DQN_seed_42_25000_steps"
        parts    = no_ext.split("_")
        try:
            # Le nombre de timesteps est toujours avant "_steps"
            steps_idx = parts.index("steps")
            return int(parts[steps_idx - 1])
        except (ValueError, IndexError):
            return 0

    # Trier par timesteps et prendre le plus récent
    checkpoints.sort(key=extract_timesteps)
    latest     = checkpoints[-1]
    steps_done = extract_timesteps(latest)

    return latest, steps_done


def load_from_checkpoint(
    algo:     str,
    env_type: str,
    seed:     int,
    env:      gym.Env,
) -> tuple:
    """
    Charge le modèle depuis le dernier checkpoint disponible.

    Retourne :
        (model, timesteps_restants)

    Lève une FileNotFoundError si aucun checkpoint n'existe.
    """
    checkpoint_path, steps_done = find_latest_checkpoint(algo, env_type, seed)

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Aucun checkpoint trouvé pour {algo} | {env_type} | seed {seed}.\n"
            f"Dossier cherché : {get_checkpoint_dir(algo, env_type)}\n"
            f"Lance d'abord un entraînement sans --resume."
        )

    steps_remaining = max(0, TOTAL_TIMESTEPS - steps_done)

    print(f"\n  Checkpoint trouvé   : {checkpoint_path}")
    print(f"  Timesteps effectués : {steps_done:,} / {TOTAL_TIMESTEPS:,}")
    print(f"  Timesteps restants  : {steps_remaining:,}")

    if steps_remaining == 0:
        print("  Entraînement déjà complet pour cette combinaison.")

    AlgoClass = ALGO_CLASSES[algo]
    model = AlgoClass.load(checkpoint_path, env=env)

    return model, steps_remaining


# ---------------------------------------------------------------------------
# Callback WandB
# ---------------------------------------------------------------------------

class WandbCallback(BaseCallback):
    """
    Callback SB3 loggant les métriques épisodiques et les losses vers WandB.

    Métriques loggées :
        - train/episode_reward  : moyenne glissante sur 10 épisodes
        - train/episode_length  : moyenne glissante sur 10 épisodes
        - train/episodes_total  : nombre total d'épisodes joués
        - train/fps             : frames per second
        - losses/*              : losses spécifiques à chaque algo
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int]   = []
        self._t_start = time.time()

    def _on_step(self) -> bool:
        # Récupération des infos épisodiques via le Monitor wrapper
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

        # Logging épisodique vers WandB
        if self._episode_rewards:
            elapsed = time.time() - self._t_start
            fps     = self.num_timesteps / max(elapsed, 1)

            wandb.log({
                "train/episode_reward": np.mean(self._episode_rewards[-10:]),
                "train/episode_length": np.mean(self._episode_lengths[-10:]),
                "train/episodes_total": len(self._episode_rewards),
                "train/fps":            fps,
                "global_step":          self.num_timesteps,
            })

        # Logging des losses spécifiques à chaque algo
        if self.model.logger and hasattr(self.model.logger, "name_to_value"):
            log_data = self.model.logger.name_to_value

            loss_keys = {
                # DQN
                "train/loss":                 "losses/td_loss",
                # PPO
                "train/policy_gradient_loss": "losses/policy_loss",
                "train/value_loss":           "losses/value_loss",
                "train/entropy_loss":         "losses/entropy",
                "train/clip_fraction":        "losses/clip_fraction",
                "train/approx_kl":            "losses/approx_kl",
                # SAC
                "train/actor_loss":           "losses/actor_loss",
                "train/critic_loss":          "losses/critic_loss",
                "train/ent_coef":             "losses/entropy_coef",
                "train/ent_coef_loss":        "losses/entropy_coef_loss",
            }

            losses_to_log = {
                wandb_key: log_data[sb3_key]
                for sb3_key, wandb_key in loss_keys.items()
                if sb3_key in log_data
            }

            if losses_to_log:
                losses_to_log["global_step"] = self.num_timesteps
                wandb.log(losses_to_log)

        return True


# ---------------------------------------------------------------------------
# Construction de l'environnement et du modèle
# ---------------------------------------------------------------------------

def make_env(env_id: str, seed: int) -> gym.Env:
    """Crée un environnement Gymnasium wrappé avec Monitor."""
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def build_model(algo: str, env: gym.Env, seed: int, verbose: int = 0):
    """Instancie un nouveau modèle SB3 avec les hyperparamètres définis."""
    params = {k: v for k, v in HYPERPARAMS[algo].items() if k != "policy"}
    policy = HYPERPARAMS[algo]["policy"]
    return ALGO_CLASSES[algo](
        policy=policy, env=env, seed=seed, verbose=verbose, **params
    )


# ---------------------------------------------------------------------------
# Boucle d'entraînement principale
# ---------------------------------------------------------------------------

def run_experiment(
    algo:            str,
    env_type:        str,
    seed:            int,
    total_timesteps: int  = TOTAL_TIMESTEPS,
    wandb_enabled:   bool = True,
    wandb_entity:    str  = None,
    resume:          bool = False,
) -> None:
    """Lance (ou reprend) un entraînement pour une combinaison algo/env/seed."""

    # Validation de la combinaison
    if env_type not in VALID_COMBINATIONS[algo]:
        raise ValueError(
            f"{algo} n'est pas compatible avec '{env_type}'. "
            f"Valides : {VALID_COMBINATIONS[algo]}"
        )

    env_id   = ENV_IDS[env_type]
    run_name = f"{algo}_{env_type}_seed{seed}"
    if resume:
        run_name += "_resumed"

    print(f"\n{'='*60}")
    print(f"  Expérience : {run_name}")
    print(f"  Mode       : {'REPRISE' if resume else 'NOUVEAU'}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"{'='*60}\n")

    # Seeding global
    set_global_seed(seed)

    # Création de l'environnement
    env = make_env(env_id, seed)

    # Modèle : nouveau ou reprise depuis checkpoint
    if resume:
        model, steps_remaining = load_from_checkpoint(algo, env_type, seed, env)
        if steps_remaining == 0:
            print("  Rien à faire, entraînement déjà complet.")
            env.close()
            return
    else:
        model           = build_model(algo, env, seed)
        steps_remaining = total_timesteps

    # Initialisation WandB
    if wandb_enabled:
        wandb.init(
            project=WANDB_PROJECT,
            entity=wandb_entity,
            name=run_name,
            group=f"{algo}_{env_type}",
            tags=[algo, env_type, f"seed_{seed}", "resumed" if resume else "fresh"],
            config={
                "algo":            algo,
                "env_id":          env_id,
                "env_type":        env_type,
                "seed":            seed,
                "total_timesteps": total_timesteps,
                "resumed":         resume,
                **HYPERPARAMS[algo],
            },
            reinit=True,          # indispensable pour plusieurs runs dans le même process
            sync_tensorboard=False,
        )

    # Construction des callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=get_checkpoint_dir(algo, env_type),
        name_prefix=get_checkpoint_prefix(algo, seed),
        verbose=1,
    )

    callbacks = [checkpoint_callback]
    if wandb_enabled:
        callbacks.append(WandbCallback())

    # Entraînement
    t_start = time.time()
    model.learn(
        total_timesteps=steps_remaining,
        callback=CallbackList(callbacks),
        reset_num_timesteps=not resume,  # False = continue le compteur de timesteps
        progress_bar=True,
    )
    elapsed = time.time() - t_start
    print(f"\n  Entraînement terminé en {elapsed:.1f}s")

    # Sauvegarde finale
    save_dir   = f"models/{algo}_{env_type}"
    os.makedirs(save_dir, exist_ok=True)
    model_path = f"{save_dir}/seed_{seed}_final"
    model.save(model_path)
    print(f"  Modèle final sauvegardé : {model_path}.zip")

    if wandb_enabled:
        wandb.log({"train/total_training_time_s": elapsed})
        wandb.finish()

    env.close()


# ---------------------------------------------------------------------------
# Évaluation post-entraînement
# ---------------------------------------------------------------------------

def evaluate_model(
    algo:            str,
    env_type:        str,
    seed:            int,
    n_eval_episodes: int  = 20,
    use_checkpoint:  bool = False,
) -> dict:
    """
    Évalue un modèle sur n épisodes en mode déterministe.

    Par défaut : charge le modèle final (seed_{seed}_final.zip).
    Avec --use-checkpoint : charge le dernier checkpoint disponible.
    """
    from stable_baselines3.common.evaluation import evaluate_policy

    AlgoClass = ALGO_CLASSES[algo]

    if use_checkpoint:
        checkpoint_path, steps_done = find_latest_checkpoint(algo, env_type, seed)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Aucun checkpoint trouvé pour {algo}/{env_type}/seed{seed}"
            )
        model_path = checkpoint_path
        print(f"  Évaluation depuis checkpoint ({steps_done:,} steps) : {model_path}")
    else:
        model_path = f"models/{algo}_{env_type}/seed_{seed}_final.zip"
        print(f"  Évaluation depuis modèle final : {model_path}")

    model    = AlgoClass.load(model_path)
    eval_env = Monitor(gym.make(ENV_IDS[env_type]))
    eval_env.reset(seed=seed + 10_000)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    print(f"\n  {algo} | {env_type} | seed {seed}")
    print(f"  Mean reward : {mean_reward:.2f} ± {std_reward:.2f}")

    eval_env.close()
    return {
        "algo":        algo,
        "env_type":    env_type,
        "seed":        seed,
        "mean_reward": mean_reward,
        "std_reward":  std_reward,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RL : DQN / PPO / SAC sur LunarLander"
    )
    parser.add_argument("--algo",      type=str, required=True, choices=["DQN", "PPO", "SAC"])
    parser.add_argument("--env",       type=str, required=True, choices=["discrete", "continuous"])
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--entity",    type=str, default=None, help="Entité WandB")
    parser.add_argument("--no-wandb",  action="store_true",    help="Désactiver WandB")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre depuis le dernier checkpoint disponible",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Évaluer sans ré-entraîner",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="(avec --eval-only) Évaluer depuis le dernier checkpoint plutôt que le modèle final",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.eval_only:
        evaluate_model(
            algo=args.algo,
            env_type=args.env,
            seed=args.seed,
            use_checkpoint=args.use_checkpoint,
        )
    else:
        run_experiment(
            algo=args.algo,
            env_type=args.env,
            seed=args.seed,
            total_timesteps=args.timesteps,
            wandb_enabled=not args.no_wandb,
            wandb_entity=args.entity,
            resume=args.resume,
        )