#!/bin/bash

GPU=$1                    # GPU ID
SEED=${2:-0}              # Random seed, default 0

DATASET="domainnet"
DATA="${COOP_DATASET}"    # Dataset root path
cfg_file="domainnet_dir_alpha0.3"
trainer=FedDnC
model=FedDnC
SUBSAMPLE_CLASSES="local"
USEALL=True
BETA=0.3                  # Dirichlet alpha
SPLIT_CLIENT=True         # Split each domain into multiple clients
IMBALANCE_TRAIN=True      # Enable label imbalance

DIR=output/${DATASET}/${trainer}/${cfg_file}/seed${SEED}
if [ ! -d "$DIR" ]; then
  echo "Error: Base model not found at ${BASE_DIR}"
  exit 1
fi

# Find model weights (best_model.pt or last_model.pt)
MODEL_PATH=""
if [ -f "${DIR}/best_model.pt" ]; then
  MODEL_PATH="${DIR}/best_model.pt"
  echo "Using best model: ${MODEL_PATH}"
elif [ -f "${DIR}/last_model.pt" ]; then
  MODEL_PATH="${DIR}/last_model.pt"
  echo "Using last model: ${MODEL_PATH}"
else
  echo "Error: Model weights not found"
  exit 1
fi
mkdir -p ${DIR}

echo "Training ${DATASET}..."
echo "Config: GPU=${GPU}, Seed=${SEED}"
echo "Params: 25 rounds, 30 clients, 25% participation, alpha=0.3"
echo "Output: ${DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
  --root ${DATA} \
  --output-dir ${DIR} \
  --seed ${SEED} \
  --model ${model} \
  --trainer ${trainer} \
  --config-file configs/trainers/${trainer}/${cfg_file}.yaml \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --eval-only \
  --model-dir ${MODEL_PATH} \
  DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
  DATASET.USEALL ${USEALL} \
  DATASET.BETA ${BETA} \
  DATASET.SPLIT_CLIENT ${SPLIT_CLIENT} \
  DATASET.IMBALANCE_TRAIN ${IMBALANCE_TRAIN}
  

echo "Training completed! Results saved to ${DIR}"
