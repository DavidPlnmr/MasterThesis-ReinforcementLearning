"""
Script d'évaluation finale multi-seeds pour LunarLander.
=========================================================
Utilisation :
    python evaluate.py --algo DQN --env discrete --train-timesteps 500000 --eval-episodes 20
    python evaluate.py --algo PPO --env discrete --train-timesteps 500000 --eval-episodes 20
    python evaluate.py --algo PPO --env continuous --train-timesteps 500000 --eval-episodes 20
    python evaluate.py --algo SAC --env continuous --train-timesteps 500000 --eval-episodes 20

Fonctionnalités :
    - Charge les meilleurs hyperparamètres depuis models/<ALGO>_<ENV>/best_params.txt
    - Entraîne sur EVAL_SEEDS seeds indépendants
    - Checkpoints toutes les 10k steps → reprise automatique si crash SLURM
    - Log complet dans WandB (courbes d'entraînement + évaluation finale)
    - Résumé sauvegardé dans models/<ALGO>_<ENV>/eval_summary.txt
"""

import argparse
import ast
import os
import random
import warnings

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

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

EVAL_SEEDS = [42, 123, 456, 789, 1337]

# ---------------------------------------------------------------------------
# Seeding global
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Chargement des meilleurs hyperparamètres
# ---------------------------------------------------------------------------

def load_best_params(algo: str, env_type: str) -> dict:
    """
    Lit models/<algo>_<env_type>/best_params.txt et retourne un dict de params.
    Format attendu (généré par tune.py) :
        Score: 250.3
        Params:
          learning_rate: 0.0003
          gamma: 0.99
          ...
    """
    path = os.path.join("models", f"{algo}_{env_type}", "best_params.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Lance d'abord : python tune.py --algo {algo} --env {env_type}"
        )

    params = {}
    best_score = None
    in_params_section = False

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Score:"):
                best_score = float(line.split(":", 1)[1].strip())
            elif line.startswith("Params:"):
                in_params_section = True
            elif in_params_section and line.startswith("  "):
                key, _, val_str = line.strip().partition(": ")
                try:
                    val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val = val_str   # garde la string telle quelle (ex: "auto", "MlpPolicy")
                params[key] = val

    if not params:
        raise ValueError(f"Aucun paramètre trouvé dans {path}")

    print(f"  Params chargés depuis {path}")
    print(f"  Score Optuna : {best_score:.2f}")
    return params


# ---------------------------------------------------------------------------
# Callback WandB — métriques épisodes + algo-spécifiques
# ---------------------------------------------------------------------------

class EvalMetricsCallback(BaseCallback):
    """
    Logue dans WandB à chaque step :
      - rollout/ep_rew_mean, ep_rew_std, ep_len_mean
      - dqn/exploration_rate (DQN uniquement)
      - sac/entropy_coef     (SAC uniquement)
    """
    def __init__(self, run, verbose: int = 0):
        super().__init__(verbose)
        self.run = run

    def _on_step(self) -> bool:
        buf = self.model.ep_info_buffer
        if len(buf) > 0:
            rewards = [ep["r"] for ep in buf]
            self.run.log({
                "rollout/ep_rew_mean": np.mean(rewards),
                "rollout/ep_rew_std":  np.std(rewards),
                "rollout/ep_len_mean": np.mean([ep["l"] for ep in buf]),
            }, step=self.num_timesteps)

        # DQN — décroissance epsilon
        if hasattr(self.model, "exploration_rate"):
            self.run.log({
                "dqn/exploration_rate": self.model.exploration_rate,
            }, step=self.num_timesteps)

        # SAC — entropie adaptative
        if hasattr(self.model, "log_ent_coef"):
            try:
                ent_coef = self.model.ent_coef_tensor.item()
            except AttributeError:
                ent_coef = float(torch.exp(self.model.log_ent_coef).detach().cpu())
            self.run.log({
                "sac/entropy_coef": ent_coef,
            }, step=self.num_timesteps)

        return True


# ---------------------------------------------------------------------------
# Entraînement d'un run (algo + seed) avec reprise sur checkpoint
# ---------------------------------------------------------------------------

def train_run(
    algo_name:       str,
    env_type:        str,
    env_id:          str,
    params:          dict,
    seed:            int,
    train_timesteps: int,
    eval_episodes:   int,
    run:             "wandb.sdk.wandb_run.Run",
) -> float:
    """
    Entraîne le modèle pour un seed donné et retourne mean_reward final.
    Reprend automatiquement depuis le dernier checkpoint si disponible.
    """
    algo_class = ALGO_CLASSES[algo_name]
    ckpt_dir   = os.path.join(
        "checkpoints", f"{algo_name}_{env_type}", f"seed_{seed}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    env = Monitor(gym.make(env_id))

    # ── Recherche du dernier checkpoint ───────────────────────────────────
    existing = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".zip") and f.startswith(f"{algo_name}_")
    ])

    model      = None
    steps_done = 0

    if existing:
        latest_ckpt = os.path.join(ckpt_dir, existing[-1])
        print(f"    → Checkpoint trouvé : {latest_ckpt}")
        try:
            model      = algo_class.load(latest_ckpt, env=env, device="auto")
            steps_done = model.num_timesteps
            print(f"    → Reprise à {steps_done}/{train_timesteps} steps")
        except Exception as e:
            warnings.warn(f"Checkpoint corrompu ({e}), repart de zéro.")
            model = None

    if model is None:
        model_kwargs = {k: v for k, v in params.items() if k != "policy"}
        model = algo_class(
            env=env,
            seed=seed,
            device="auto",
            verbose=0,
            policy=params.get("policy", "MlpPolicy"),
            **model_kwargs,
        )

    # ── Entraînement (si pas déjà complet) ────────────────────────────────
    remaining = train_timesteps - steps_done
    if remaining <= 0:
        print(f"    → Entraînement déjà complet ({steps_done} steps).")
    else:
        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=ckpt_dir,
            name_prefix=algo_name,
            save_replay_buffer=True,    # important pour DQN/SAC (off-policy)
            save_vecnormalize=False,
        )
        metrics_cb = EvalMetricsCallback(run=run)

        model.learn(
            total_timesteps=remaining,
            callback=[checkpoint_cb, metrics_cb],
            reset_num_timesteps=False,  # conserve le compteur global de steps
        )

    # ── Évaluation finale déterministe ────────────────────────────────────
    mean_reward, std_reward = evaluate_policy(
        model, env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )
    run.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward":  std_reward,
        "eval/seed":        seed,
    })

    env.close()
    return mean_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",            type=str, required=True,
                        choices=["DQN", "PPO", "SAC"])
    parser.add_argument("--env",             type=str, required=True,
                        choices=["discrete", "continuous"])
    parser.add_argument("--train-timesteps", type=int, default=500_000,
                        help="Timesteps d'entraînement par seed")
    parser.add_argument("--eval-episodes",   type=int, default=20,
                        help="Épisodes d'évaluation finale par seed")
    parser.add_argument("--wandb-project",   type=str,
                        default="rl-lunarlander-eval")
    args = parser.parse_args()

    if args.env not in VALID_COMBINATIONS[args.algo]:
        raise ValueError(
            f"Combinaison invalide : {args.algo} + env={args.env}. "
            f"Valides : {VALID_COMBINATIONS}"
        )

    env_id = ENV_IDS[args.env]
    key    = f"{args.algo}_{args.env}"

    # ── Device info ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Évaluation : {args.algo} | {args.env} | {len(EVAL_SEEDS)} seeds")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA {torch.version.cuda} — sm_61 : {'sm_61' in torch.cuda.get_arch_list()}")
    else:
        print("  CPU uniquement.")
    print()

    # ── Chargement des meilleurs params ────────────────────────────────────
    params = load_best_params(args.algo, args.env)
    print()

    # ── Boucle seeds ──────────────────────────────────────────────────────
    rewards = []

    for seed in EVAL_SEEDS:
        set_global_seed(seed)
        print(f"  Seed {seed}...")

        if wandb.run is not None:
            wandb.finish()

        run = wandb.init(
            project=args.wandb_project,
            group=key,              # ex: "DQN_discrete"
            name=f"seed_{seed}",
            config={
                **params,
                "algo":            args.algo,
                "env":             env_id,
                "seed":            seed,
                "train_timesteps": args.train_timesteps,
                "eval_episodes":   args.eval_episodes,
            },
            reinit=True,
            dir=os.environ.get("WANDB_DIR", "."),
        )

        mean_reward = train_run(
            algo_name=args.algo,
            env_type=args.env,
            env_id=env_id,
            params=params,
            seed=seed,
            train_timesteps=args.train_timesteps,
            eval_episodes=args.eval_episodes,
            run=run,
        )

        rewards.append(mean_reward)
        print(f"    → mean_reward = {mean_reward:.2f}")
        wandb.finish()

    # ── Résumé final ───────────────────────────────────────────────────────
    arr    = np.array(rewards)
    solved = int(np.sum(arr >= 200))

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ — {key}")
    print(f"{'='*60}")
    print(f"  Mean  : {arr.mean():.2f}")
    print(f"  Std   : {arr.std():.2f}")
    print(f"  Min   : {arr.min():.2f}")
    print(f"  Max   : {arr.max():.2f}")
    print(f"  Résolu: {solved}/{len(EVAL_SEEDS)} seeds (reward ≥ 200)")
    print(f"{'='*60}\n")

    # ── Sauvegarde ─────────────────────────────────────────────────────────
    out_dir = os.path.join("models", key)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Algo : {args.algo}\n")
        f.write(f"Env  : {args.env}\n")
        f.write(f"Seeds: {EVAL_SEEDS}\n\n")
        f.write(f"Mean  : {arr.mean():.2f}\n")
        f.write(f"Std   : {arr.std():.2f}\n")
        f.write(f"Min   : {arr.min():.2f}\n")
        f.write(f"Max   : {arr.max():.2f}\n")
        f.write(f"Résolu: {solved}/{len(EVAL_SEEDS)} seeds\n\n")
        f.write("Détail par seed:\n")
        for s, r in zip(EVAL_SEEDS, rewards):
            f.write(f"  seed {s}: {r:.2f}\n")

    print(f"  Résumé sauvegardé dans : {summary_path}")


if __name__ == "__main__":
    main()