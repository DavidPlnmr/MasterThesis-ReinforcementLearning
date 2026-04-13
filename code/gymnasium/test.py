import os
import time
import gymnasium as gym
from stable_baselines3 import DQN, PPO, SAC

def test_model(model_dir, model_name, algo_class, env_id):
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Modèle introuvable : {model_path}")
        return
         
    print(f"\n--- Test de {model_path} sur {env_id} ---")
    model = algo_class.load(model_path)
    env = gym.make(env_id, render_mode="human")
    
    obs, info = env.reset()
    done = truncated = False
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(0.01) # Ralentir la simulation si besoin
        
    env.close()
    print("Épisode terminé.\n")

if __name__ == "__main__":
    # Assurez-vous d'être dans le dossier code/gymnasium ou d'adapter le chemin
    base_dir = "models"
    
    # Configurations des différents modèles à tester
    configs = [
        {"dir": "DQN_discrete", "algo": DQN, "env": "LunarLander-v3"},
        {"dir": "PPO_discrete", "algo": PPO, "env": "LunarLander-v3"},
        {"dir": "PPO_continuous", "algo": PPO, "env": "LunarLanderContinuous-v3"},
        {"dir": "SAC_continuous", "algo": SAC, "env": "LunarLanderContinuous-v3"},
    ]
    
    for config in configs:
        model_dir = os.path.join(base_dir, config["dir"])
        # On va tester le modèle final pour chaque algorithme (seed_42_final.zip)
        model_name = "seed_42_final.zip"
        test_model(model_dir, model_name, config["algo"], config["env"])
