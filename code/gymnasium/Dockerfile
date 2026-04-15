FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        swig \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    "numpy<2.0" \
    "stable-baselines3[extra]==2.3.2" \
    "gymnasium[box2d]==0.29.1" \
    optuna==3.6.1 \
    "optuna-integration[wandb]==3.6.0" \
    "wandb==0.17.0" \
    tensorboard==2.16.2

CMD ["/bin/bash"]