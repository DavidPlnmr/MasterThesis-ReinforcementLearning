# Utiliser l'image officielle de PyTorch avec CUDA comme base
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système requises (swig et build-essential sont nécessaires pour Box2D / LunarLander)
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --no-cache-dir --upgrade pip


# Installer les dépendances Python pour l'apprentissage par renforcement
# Vous pouvez aussi utiliser un fichier requirements.txt si vous préférez : COPY requirements.txt . && pip install -r requirements.txt
RUN pip install --no-cache-dir \
    "stable-baselines3[extra]" \
    "gymnasium[box2d]" \
    wandb \
    tensorboard \
    optuna \
    "optuna-integration[wandb]"

# Command par défaut
CMD ["/bin/bash"]
