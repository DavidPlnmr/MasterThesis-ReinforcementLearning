import gymnasium as gym
from stable_baselines3 import PPO

# Charger le modèle entraîné
model = PPO.load("ppo_car_racing.zip")

# Créer l'environnement avec rendu
env = gym.make("CarRacing-v3", render_mode="human")

# Reset
obs, info = env.reset()

for i in range(10):

    done = False

    while not done:
        # Prédire l'action
        action, _ = model.predict(obs, deterministic=True)

        # Print l'action choisie
        print(f"Action choisie: {action}")

        # Avancer dans l'environnement
        obs, reward, terminated, truncated, info = env.step(action)

        # Vérifier fin d'épisode
        done = terminated or truncated

    if done:
        obs, info = env.reset()
        done = False
        
env.close()
