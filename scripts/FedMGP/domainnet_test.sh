#!/bin/bash

GPU=$1                    # GPU ID
SEED=${2:-0}              # Random seed, default 0
MODEL_DIR=$3              # Model directory path (required)

if [ -z "$MODEL_DIR" ]; then
    echo "Error: Model directory path required as 3rd argument"
    exit 1
fi

DATASET="domainnet"
DATA="${COOP_DATASET}"    # Dataset root path
cfg_file="domainnet_dir_alpha0.3"
trainer=FedMGP
model=fedavg
SUBSAMPLE_CLASSES="local"
USEALL=True
BETA=0.3                  # Dirichlet alpha
SPLIT_CLIENT=True         # Split each domain into multiple clients
IMBALANCE_TRAIN=True      # Enable label imbalance

TEST_DIR=inference_results/logs/${DATASET}/${trainer}/${cfg_file}/seed${SEED}

mkdir -p ${TEST_DIR}

echo "Testing ${DATASET}..."
echo "Config: GPU=${GPU}, Seed=${SEED}"
echo "Model: ${MODEL_DIR}"
echo "Output: ${TEST_DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
  --root ${DATA} \
  --output-dir ${TEST_DIR} \
  --seed ${SEED} \
  --model ${model} \
  --trainer ${trainer} \
  --config-file configs/trainers/${trainer}/${cfg_file}.yaml \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --eval-only \
  --model-dir ${MODEL_DIR}/last_model.pt \
  DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
  DATASET.USEALL ${USEALL} \
  DATASET.BETA ${BETA} \
  DATASET.SPLIT_CLIENT ${SPLIT_CLIENT} \
  DATASET.IMBALANCE_TRAIN ${IMBALANCE_TRAIN}

echo "Testing completed! Results saved to ${TEST_DIR}"
