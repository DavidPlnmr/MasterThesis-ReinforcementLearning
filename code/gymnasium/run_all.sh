#!/bin/bash
# =============================================================================
# run_all.sh — Lance toutes les combinaisons algo × env × seed
# =============================================================================
# Usage :
#   chmod +x run_all.sh
#   ./run_all.sh              # toutes les seeds
#   ./run_all.sh --fast       # seed unique (42) pour test rapide
# =============================================================================

SEEDS=(42 123 456 789 1337)
TIMESTEPS=500000
FAST_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --fast) FAST_MODE=true ;;
    esac
done

if [ "$FAST_MODE" = true ]; then
    SEEDS=(42)
    TIMESTEPS=100000
    echo ">>> Mode rapide : 1 seed, 100k timesteps"
fi

echo "=================================================="
echo "  Benchmark LunarLander — $(date)"
echo "  Seeds       : ${SEEDS[*]}"
echo "  Timesteps   : $TIMESTEPS"
echo "=================================================="

# Combinaisons valides
#   DQN  -> discret seulement
#   PPO  -> discret + continu
#   SAC  -> continu seulement

run_experiment() {
    local ALGO=$1
    local ENV=$2
    local SEED=$3
    echo ""
    echo ">>> $ALGO | $ENV | seed=$SEED"
    python train.py --algo "$ALGO" --env "$ENV" --seed "$SEED" --timesteps "$TIMESTEPS"
    if [ $? -ne 0 ]; then
        echo "ERREUR : $ALGO $ENV seed=$SEED a échoué."
        exit 1
    fi
}

for SEED in "${SEEDS[@]}"; do

    # --- Environnement DISCRET ---
    run_experiment DQN discrete $SEED
    run_experiment PPO discrete $SEED

    # --- Environnement CONTINU ---
    run_experiment PPO continuous $SEED
    run_experiment SAC continuous $SEED

done

echo ""
echo "=================================================="
echo "  Toutes les expériences terminées."
echo "=================================================="