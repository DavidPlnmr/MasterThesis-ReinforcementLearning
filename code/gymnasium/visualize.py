"""Visualisation de modèles Stable-Baselines3 sauvegardés.

Le dossier attendu est typiquement :

models_saved/
  PPO_discrete/
    seed_42/
      .../checkpoint_100000_steps.zip
    seed_123/
      .../checkpoint_100000_steps.zip
  PPO_continuous/
  SAC_continuous/
  DQN_discrete/

Le script :
- parcourt automatiquement tous les sous-dossiers d'algorithme,
- détecte la seed,
- récupère le dernier checkpoint .zip trouvé,
- charge le modèle SB3,
- lance 5 épisodes par modèle,
- enregistre une vidéo par épisode dans un dossier dédié.

Usage :
    python visualize.py --models-root models_saved --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo


ENV_IDS = {
    "discrete": "LunarLander-v2",
    "continuous": "LunarLanderContinuous-v2",
}

ALGO_CLASSES = {
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
}

VALID_COMBINATIONS = {
    "DQN_discrete": ("DQN", "discrete"),
    "PPO_discrete": ("PPO", "discrete"),
    "PPO_continuous": ("PPO", "continuous"),
    "SAC_continuous": ("SAC", "continuous"),
}


@dataclass
class ModelRunResult:
    combination: str
    algorithm: str
    env_type: str
    seed: int
    model_path: str
    env_id: str
    n_eval_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float
    std_length: float
    episode_rewards: list[float]
    episode_lengths: list[int]
    video_dir: str


def parse_seed(folder_name: str) -> int:
    match = re.search(r"seed_(\d+)", folder_name)
    if not match:
        raise ValueError(f"Impossible d'extraire la seed depuis: {folder_name}")
    return int(match.group(1))


def find_latest_zip(seed_dir: Path) -> Path:
    """Retourne le checkpoint .zip le plus récent dans un dossier seed."""
    zip_files = list(seed_dir.rglob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"Aucun fichier .zip trouvé dans {seed_dir}")

    def sort_key(path: Path) -> tuple[int, float]:
        # Priorité au nombre de steps si présent dans le nom, sinon à la date.
        m = re.search(r"_(\d+)_steps\.zip$", path.name)
        steps = int(m.group(1)) if m else -1
        return (steps, path.stat().st_mtime)

    return max(zip_files, key=sort_key)


def make_env(env_type: str, seed: int | None = None, render_mode: str = "rgb_array"):
    env_id = ENV_IDS[env_type]
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def load_model(algorithm: str, model_path: Path, env):
    algo_class = ALGO_CLASSES[algorithm]
    return algo_class.load(str(model_path), env=env, device="cpu")


def rollout_episode(model, env, deterministic: bool = True) -> tuple[float, int]:
    """Joue un épisode complet et retourne (reward_total, longueur)."""
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    length = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        # Convertir l'action en type natif Python (évite les incompatibilités Box2D)
        action = np.asarray(action)
        if action.ndim == 0 or action.size == 1:
            # Action scalaire (discrète) — convertir en int
            action = int(action.flat[0])
        else:
            # Action vectorielle (continue) — convertir en numpy array de floats
            # (Box2D accepte les numpy arrays mais pas les tuples/listes de numpy scalars)
            action = action.astype(np.float64, copy=False)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        length += 1

    return total_reward, length


def evaluate_saved_model(
    algorithm: str,
    env_type: str,
    model_path: Path,
    seed: int,
    video_root: Path,
    n_eval_episodes: int = 5,
    deterministic: bool = True,
) -> ModelRunResult | None:
    env_id = ENV_IDS[env_type]
    video_dir = video_root / f"{algorithm}_{env_type}" / f"seed_{seed}"
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Créer l'env sans rendu d'abord pour l'évaluation
        eval_env = make_env(env_type, seed=seed + 10_000, render_mode=None)
        
        model = load_model(algorithm, model_path, eval_env)
        
        # Utiliser evaluate_policy de SB3 qui gère les conversions correctement
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
        )
        
        eval_env.close()
        
        # Créer des vidéos par la suite (séparément pour éviter les incompatibilités)
        video_count = 0
        for ep_idx in range(n_eval_episodes):
            try:
                # Créer un nouvel env juste pour l'enregistrement
                record_env = make_env(env_type, seed=seed + 10_000 + ep_idx, render_mode="rgb_array")
                record_env = RecordVideo(
                    record_env,
                    video_folder=str(video_dir),
                    name_prefix=f"{algorithm}_{env_type}_seed_{seed}_ep_{ep_idx}",
                    episode_trigger=lambda episode_id: episode_id == 0,
                    disable_logger=True,
                )
                
                # Charger une copie fraîche du modèle
                model_copy = load_model(algorithm, model_path, record_env)
                # Jouer l'épisode
                obs, _ = record_env.reset()
                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = model_copy.predict(obs, deterministic=deterministic)
                    obs, reward, done, truncated, _ = record_env.step(action)
                record_env.close()
                video_count += 1
            except Exception as e:
                # Silencieusement échouer sur les vidéos (pas critique)
                try:
                    record_env.close()
                except:
                    pass
                continue
        
        if video_count > 0:
            print(f"    ✓ {video_count}/{n_eval_episodes} vidéo(s) enregistrée(s).")
        
        return ModelRunResult(
            combination=f"{algorithm}_{env_type}",
            algorithm=algorithm,
            env_type=env_type,
            seed=seed,
            model_path=str(model_path),
            env_id=env_id,
            n_eval_episodes=len(episode_rewards),
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            min_reward=float(np.min(episode_rewards)),
            max_reward=float(np.max(episode_rewards)),
            mean_length=float(np.mean(episode_lengths)),
            std_length=float(np.std(episode_lengths)),
            episode_rewards=[float(x) for x in episode_rewards],
            episode_lengths=[int(x) for x in episode_lengths],
            video_dir=str(video_dir),
        )
    except Exception as e:
        print(f"    ✗ Erreur critique lors de l'évaluation : {type(e).__name__}: {str(e)[:100]}")
        return None


def iter_model_folders(models_root: Path) -> Iterable[tuple[str, Path]]:
    for child in sorted(models_root.iterdir()):
        if child.is_dir() and child.name in VALID_COMBINATIONS:
            yield child.name, child


def discover_runs(models_root: Path):
    """Produit les couples (combinaison, seed, checkpoint)."""
    for combo_name, combo_dir in iter_model_folders(models_root):
        algorithm, env_type = VALID_COMBINATIONS[combo_name]
        for seed_dir in sorted(combo_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = parse_seed(seed_dir.name)
            ckpt = find_latest_zip(seed_dir)
            yield {
                "combination": combo_name,
                "algorithm": algorithm,
                "env_type": env_type,
                "seed": seed,
                "checkpoint": ckpt,
            }


def print_result(result: ModelRunResult) -> None:
    print("=" * 88)
    print(f"{result.combination} | seed {result.seed}")
    print(f"Checkpoint : {result.model_path}")
    print(f"Videos     : {result.video_dir}")
    print(f"Env        : {result.env_id}")
    print(
        f"Reward     : {result.mean_reward:.2f} ± {result.std_reward:.2f} "
        f"[min={result.min_reward:.2f}, max={result.max_reward:.2f}]"
    )
    print(
        f"Length     : {result.mean_length:.1f} ± {result.std_length:.1f} "
        f"sur {result.n_eval_episodes} épisodes"
    )
    print(f"Episodes    : {result.episode_rewards}")


def build_summary(results: list[ModelRunResult]) -> dict:
    grouped: dict[str, list[ModelRunResult]] = {}
    for result in results:
        grouped.setdefault(result.combination, []).append(result)

    summary: dict[str, dict] = {}
    for combo, combo_results in grouped.items():
        rewards = np.array([r.mean_reward for r in combo_results], dtype=float)
        lengths = np.array([r.mean_length for r in combo_results], dtype=float)
        summary[combo] = {
            "n_models": len(combo_results),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "details": [asdict(r) for r in combo_results],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-root",
        type=str,
        default="models_saved",
        help="Dossier racine contenant PPO_discrete, PPO_continuous, SAC_continuous, DQN_discrete.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Nombre d'épisodes à jouer et à enregistrer par modèle.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Politique déterministe pendant la lecture des vidéos (par défaut activée).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Chemin optionnel pour sauvegarder le résumé en JSON.",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="videos",
        help="Dossier racine où enregistrer les vidéos.",
    )
    args = parser.parse_args()

    models_root = Path(args.models_root).expanduser().resolve()
    if not models_root.exists():
        raise FileNotFoundError(f"Dossier introuvable : {models_root}")

    video_root = Path(args.video_root).expanduser().resolve()
    video_root.mkdir(parents=True, exist_ok=True)

    discovered = list(discover_runs(models_root))
    if not discovered:
        raise RuntimeError(
            f"Aucun modèle trouvé dans {models_root}. "
            f"Attendu: {', '.join(VALID_COMBINATIONS.keys())}"
        )

    results: list[ModelRunResult] = []

    for item in discovered:
        result = evaluate_saved_model(
            algorithm=item["algorithm"],
            env_type=item["env_type"],
            model_path=item["checkpoint"],
            seed=item["seed"],
            video_root=video_root,
            n_eval_episodes=args.episodes,
            deterministic=args.deterministic,
        )
        if result is not None:
            results.append(result)
            print_result(result)
        else:
            print(f"Modèle ignoré : {item['algorithm']}_{item['env_type']}/seed_{item['seed']}")

    summary = build_summary(results)

    print("\n" + "#" * 88)
    print("RÉSUMÉ GLOBAL")
    print("#" * 88)
    for combo, data in summary.items():
        print(
            f"{combo:18s} | n={data['n_models']} | "
            f"reward={data['mean_reward']:.2f} ± {data['std_reward']:.2f} | "
            f"len={data['mean_length']:.1f} ± {data['std_length']:.1f}"
        )

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nRésumé sauvegardé dans : {output_path}")


if __name__ == "__main__":
    main()
