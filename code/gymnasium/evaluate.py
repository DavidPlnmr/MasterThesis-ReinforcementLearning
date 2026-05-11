"""
Script d'évaluation finale multi-seeds pour LunarLander.
=========================================================
Utilisation :
    python evaluate.py --algo DQN --env discrete --train-timesteps 500000 --eval-episodes 20
    python evaluate.py --algo PPO --env discrete --train-timesteps 500000 --eval-episodes 20
    python evaluate.py --algo PPO --env continuous --train-timesteps 500000 --eval-episodes 20
    python evaluate.py --algo SAC --env continuous --train-timesteps 500000 --eval-episodes 20

Reprise automatique :
    - Si une seed est entièrement terminée (résultat dans eval_results.json),
      elle est sautée sans relancer ni WandB ni entraînement.
    - Si une seed est partiellement entraînée (checkpoint .zip présent),
      l'entraînement reprend depuis le dernier checkpoint.
    - Si aucune seed n'a encore été traitée, tout repart de zéro.
"""

import argparse
import ast
import json
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

NET_ARCH_MAP = {
    "small":  [64, 64],
    "medium": [256, 256],
    "large":  [400, 300],
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
# Persistance des résultats par seed
# ---------------------------------------------------------------------------

def results_path(algo: str, env_type: str) -> str:
    """Chemin du fichier JSON qui trace les résultats par seed."""
    return os.path.join("models", f"{algo}_{env_type}", "eval_results.json")


def load_completed_results(algo: str, env_type: str) -> dict:
    """
    Charge le dict des seeds déjà complétées.
    Format : { "42": {"mean_reward": 250.3, "std_reward": 12.1}, ... }
    Retourne {} si le fichier n'existe pas encore.
    """
    path = results_path(algo, env_type)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_seed_result(
    algo:        str,
    env_type:    str,
    seed:        int,
    mean_reward: float,
    std_reward:  float,
    mean_length: float = 0.0,
    std_length:  float = 0.0,
) -> None:
    """
    Ajoute ou met à jour le résultat d'une seed dans eval_results.json.
    Écriture atomique : on recharge, on modifie, on réécrit.
    """
    path = results_path(algo, env_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    results = load_completed_results(algo, env_type)
    results[str(seed)] = {
        "mean_reward": float(mean_reward),
        "std_reward":  float(std_reward),
        "mean_length": float(mean_length),
        "std_length":  float(std_length),
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


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
          net_arch: medium
          ...
    """
    path = os.path.join("models", f"{algo}_{env_type}", "best_params.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Lance d'abord : python tune.py --algo {algo} --env {env_type}"
        )

    params      = {}
    best_score  = None
    in_params   = False

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Score:"):
                best_score = float(line.split(":", 1)[1].strip())
            elif line.startswith("Params:"):
                in_params = True
            elif in_params and line.startswith("  "):
                key, _, val_str = line.strip().partition(": ")
                try:
                    val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val = val_str
                params[key] = val

    if "net_arch" in params:
        net_arch_key = params.pop("net_arch")
        params["policy_kwargs"] = {"net_arch": NET_ARCH_MAP[net_arch_key]}

    if not params:
        raise ValueError(f"Aucun paramètre trouvé dans {path}")

    print(f"  Params chargés depuis {path}")
    if best_score is not None:
        print(f"  Score Optuna : {best_score:.2f}")
    return params


# ---------------------------------------------------------------------------
# Callback WandB — métriques épisodes + algo-spécifiques
# ---------------------------------------------------------------------------

class EvalMetricsCallback(BaseCallback):
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

        if hasattr(self.model, "exploration_rate"):
            self.run.log({
                "dqn/exploration_rate": self.model.exploration_rate,
            }, step=self.num_timesteps)

        if hasattr(self.model, "log_ent_coef"):
            try:
                ent_coef = self.model.ent_coef_tensor.item()
            except AttributeError:
                ent_coef = float(torch.exp(self.model.log_ent_coef).detach().cpu())
            self.run.log({"sac/entropy_coef": ent_coef}, step=self.num_timesteps)

        return True


# ---------------------------------------------------------------------------
# Entraînement + évaluation d'une seed
# ---------------------------------------------------------------------------

def train_and_evaluate(
    algo_name:       str,
    env_type:        str,
    env_id:          str,
    params:          dict,
    seed:            int,
    train_timesteps: int,
    eval_episodes:   int,
    run:             "wandb.sdk.wandb_run.Run",
) -> tuple[float, float, float, float]:
    """
    Entraîne le modèle pour un seed donné (avec reprise sur checkpoint)
    et retourne (mean_reward, std_reward, mean_length, std_length) de l'évaluation finale.
    """
    algo_class = ALGO_CLASSES[algo_name]
    ckpt_dir   = os.path.join(
        "checkpoints", f"{algo_name}_{env_type}", f"seed_{seed}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    train_env = Monitor(gym.make(env_id))
    train_env.reset(seed=seed)

    # ── Recherche du dernier checkpoint ───────────────────────────────────
    existing = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".zip") and f.startswith(f"{algo_name}_")
    ])

    model      = None
    steps_done = 0

    if existing:
        latest_ckpt = os.path.join(ckpt_dir, existing[-1])
        print(f"    → Checkpoint trouvé : {os.path.basename(latest_ckpt)}")
        try:
            model      = algo_class.load(latest_ckpt, env=train_env, device="cpu")
            steps_done = model.num_timesteps
            print(f"    → Reprise à {steps_done:,}/{train_timesteps:,} steps")
        except Exception as e:
            warnings.warn(f"Checkpoint corrompu ({e}), repart de zéro.")
            model = None

    if model is None:
        model_kwargs = {k: v for k, v in params.items() if k != "policy"}
        model = algo_class(
            env=train_env,
            seed=seed,
            device="cpu",
            verbose=0,
            policy=params.get("policy", "MlpPolicy"),
            **model_kwargs,
        )

    # ── Entraînement ──────────────────────────────────────────────────────
    remaining = train_timesteps - steps_done
    if remaining <= 0:
        print(f"    → Entraînement déjà complet ({steps_done:,} steps).")
    else:
        print(f"    → Entraînement : {remaining:,} steps restants...")
        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=ckpt_dir,
            name_prefix=algo_name,
            save_replay_buffer=True,
            save_vecnormalize=False,
        )
        model.learn(
            total_timesteps=remaining,
            callback=[checkpoint_cb, EvalMetricsCallback(run=run)],
            reset_num_timesteps=False,  # conserve le compteur global
        )

    # ── Évaluation finale déterministe ────────────────────────────────────
    eval_env = Monitor(gym.make(env_id))
    eval_env.reset(seed=seed + 9999)

    ep_rewards, ep_lengths = evaluate_policy(
        model, eval_env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )

    mean_reward = float(np.mean(ep_rewards))
    std_reward  = float(np.std(ep_rewards))
    mean_length = float(np.mean(ep_lengths))
    std_length  = float(np.std(ep_lengths))

    run.summary["eval/mean_reward"] = mean_reward
    run.summary["eval/std_reward"]  = std_reward
    run.summary["eval/mean_length"] = mean_length
    run.summary["eval/std_length"]  = std_length
    run.summary["eval/seed"]        = seed

    train_env.close()
    eval_env.close()
    return mean_reward, std_reward, mean_length, std_length


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",            type=str, required=True,
                        choices=["DQN", "PPO", "SAC"])
    parser.add_argument("--env",             type=str, required=True,
                        choices=["discrete", "continuous"])
    parser.add_argument("--train-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-episodes",   type=int, default=20)
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

    print(f"\n{'='*60}")
    print(f"  Évaluation : {args.algo} | {args.env} | {len(EVAL_SEEDS)} seeds")
    print(f"{'='*60}")

    # ── Chargement des seeds déjà complétées ──────────────────────────────
    completed = load_completed_results(args.algo, args.env)

    if all(str(s) in completed for s in EVAL_SEEDS):
        # Toutes les seeds sont terminées — affiche le résumé et sort immédiatement
        # sans charger les params, sans init WandB, sans toucher aux checkpoints.
        print("  ✓ Toutes les seeds sont déjà complétées.\n")
        rewards = np.array([completed[str(s)]["mean_reward"] for s in EVAL_SEEDS])
        solved  = int(np.sum(rewards >= 200))
        print(f"{'='*60}")
        print(f"  RÉSUMÉ FINAL — {key}")
        print(f"{'='*60}")
        print(f"  Mean  : {rewards.mean():.2f}")
        print(f"  Std   : {rewards.std():.2f}")
        print(f"  Min   : {rewards.min():.2f}")
        print(f"  Max   : {rewards.max():.2f}")
        print(f"  Résolu: {solved}/{len(EVAL_SEEDS)} seeds (reward ≥ 200)")
        for s in EVAL_SEEDS:
            r = completed[str(s)]
            m_len = r.get("mean_length", 0.0)
            s_len = r.get("std_length", 0.0)
            print(f"    seed {s:4d} : {r['mean_reward']:7.2f} ± {r['std_reward']:.2f}  (len: {m_len:6.1f} ± {s_len:.1f})")
        print(f"{'='*60}\n")
        return

    if completed:
        done_seeds = [int(s) for s in completed]
        print(f"  Seeds déjà complétées : {done_seeds}")
        print(f"  Seeds restantes       : "
              f"{[s for s in EVAL_SEEDS if s not in done_seeds]}")
    else:
        print("  Aucune seed complétée — démarrage complet.")
    print()

    # ── Chargement des params ──────────────────────────────────────────────
    params = load_best_params(args.algo, args.env)
    print()

    # ── Boucle seeds ──────────────────────────────────────────────────────
    for seed in EVAL_SEEDS:

        # ── Seed déjà terminée → skip ─────────────────────────────────────
        if str(seed) in completed:
            r = completed[str(seed)]
            m_len = r.get("mean_length", 0.0)
            print(f"  Seed {seed} : déjà complétée "
                  f"(mean={r['mean_reward']:.2f}, std={r['std_reward']:.2f} | len={m_len:.1f}) — skip.")
            continue

        print(f"\n  Seed {seed}...")
        set_global_seed(seed)

        # Ferme proprement un éventuel run WandB orphelin
        if wandb.run is not None:
            wandb.finish()

        run = wandb.init(
            project=args.wandb_project,
            group=key,
            name=f"{key}_seed_{seed}",
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
            settings=wandb.Settings(start_method="thread"),
        )

        try:
            mean_reward, std_reward, mean_length, std_length = train_and_evaluate(
                algo_name=args.algo,
                env_type=args.env,
                env_id=env_id,
                params=params,
                seed=seed,
                train_timesteps=args.train_timesteps,
                eval_episodes=args.eval_episodes,
                run=run,
            )
        except Exception as exc:
            warnings.warn(f"Seed {seed} échouée : {exc}")
            run.finish(exit_code=1)
            # On ne sauvegarde PAS le résultat → la seed sera relancée
            raise

        # ── Sauvegarde immédiate du résultat ──────────────────────────────
        # Fait AVANT wandb.finish() pour garantir la persistance même si
        # finish() crashe ou si le job SLURM est tué juste après.
        save_seed_result(args.algo, args.env, seed, mean_reward, std_reward, mean_length, std_length)
        print(f"    → mean_reward = {mean_reward:.2f} ± {std_reward:.2f} | len = {mean_length:.1f} ± {std_length:.1f} [sauvegardé]")

        run.finish()

    # ── Résumé final ───────────────────────────────────────────────────────
    # Recharge depuis le fichier pour inclure les seeds skipées
    final_results = load_completed_results(args.algo, args.env)

    # Vérifie que toutes les seeds sont présentes
    missing = [s for s in EVAL_SEEDS if str(s) not in final_results]
    if missing:
        print(f"\n  ⚠️  Seeds manquantes (non complétées) : {missing}")
        print("  Relancez le script pour compléter.")
        return

    rewards = np.array([final_results[str(s)]["mean_reward"] for s in EVAL_SEEDS])
    solved  = int(np.sum(rewards >= 200))

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ FINAL — {key}")
    print(f"{'='*60}")
    print(f"  Mean  : {rewards.mean():.2f}")
    print(f"  Std   : {rewards.std():.2f}")
    print(f"  Min   : {rewards.min():.2f}")
    print(f"  Max   : {rewards.max():.2f}")
    print(f"  Résolu: {solved}/{len(EVAL_SEEDS)} seeds (reward ≥ 200)")
    for s in EVAL_SEEDS:
        r = final_results[str(s)]
        m_len = r.get("mean_length", 0.0)
        s_len = r.get("std_length", 0.0)
        print(f"    seed {s:4d} : {r['mean_reward']:7.2f} ± {r['std_reward']:.2f}  (len: {m_len:6.1f} ± {s_len:.1f})")
    print(f"{'='*60}\n")

    # ── Sauvegarde du résumé texte ─────────────────────────────────────────
    out_dir      = os.path.join("models", key)
    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Algo  : {args.algo}\n")
        f.write(f"Env   : {args.env}\n")
        f.write(f"Seeds : {EVAL_SEEDS}\n\n")
        f.write(f"Mean  : {rewards.mean():.2f}\n")
        f.write(f"Std   : {rewards.std():.2f}\n")
        f.write(f"Min   : {rewards.min():.2f}\n")
        f.write(f"Max   : {rewards.max():.2f}\n")
        f.write(f"Résolu: {solved}/{len(EVAL_SEEDS)} seeds\n\n")
        f.write("Détail par seed:\n")
        for s in EVAL_SEEDS:
            r = final_results[str(s)]
            m_len = r.get("mean_length", 0.0)
            s_len = r.get("std_length", 0.0)
            f.write(f"  seed {s}: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}  (len: {m_len:.1f} ± {s_len:.1f})\n")

    print(f"  Résumé sauvegardé dans : {summary_path}")


if __name__ == "__main__":
    main()