#!/bin/bash

GPU=$1                    # GPU ID
SEED=${2:-0}              # Random seed, default 0

DATASET="office"
DATA="${COOP_DATASET}"    # Dataset root path
cfg_file="office_dir_alpha0.3"
trainer=FedMGP
model=fedavg
SUBSAMPLE_CLASSES="local"
USEALL=True
BETA=0.3                  # Dirichlet alpha
SPLIT_CLIENT=True         # Split each domain into multiple clients
IMBALANCE_TRAIN=True      # Enable label imbalance

DIR=output/${DATASET}/${trainer}/${cfg_file}/seed${SEED}

mkdir -p ${DIR}

echo "Training ${DATASET}..."
echo "Config: GPU=${GPU}, Seed=${SEED}"
echo "Params: 25 rounds, 20 clients, 50% participation, alpha=0.3"
echo "Output: ${DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
  --root ${DATA} \
  --output-dir ${DIR} \
  --seed ${SEED} \
  --model ${model} \
  --trainer ${trainer} \
  --config-file configs/trainers/${trainer}/${cfg_file}.yaml \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
  DATASET.USEALL ${USEALL} \
  DATASET.BETA ${BETA} \
  DATASET.SPLIT_CLIENT ${SPLIT_CLIENT} \
  DATASET.IMBALANCE_TRAIN ${IMBALANCE_TRAIN}

echo "Training completed! Results saved to ${DIR}"
