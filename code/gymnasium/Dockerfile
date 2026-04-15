FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
 
WORKDIR /app
 
# ── Dépendances système ─────────────────────────────────────────────────────
# swig + build-essential : requis pour compiler Box2D (LunarLander)
# git                    : requis par certains packages pip installés depuis
#                          des dépôts GitHub (e.g. sb3-contrib en snapshot)
RUN apt-get update && apt-get install -y --no-install-recommends \
        swig \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*
 
# ── Dépendances Python ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        "stable-baselines3[extra]==2.3.2" \
        "gymnasium[box2d]==0.29.1" \
        optuna==3.6.1 \
        "optuna-integration[wandb]==3.6.1" \
        wandb==0.17.0 \
        tensorboard==2.16.2
 
CMD ["/bin/bash"]