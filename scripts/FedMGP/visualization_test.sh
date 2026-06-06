#!/bin/bash

# FedMGP attention visualization test script
# Usage: ./visualization_test.sh <GPU> <DATASET> <SUBSAMPLE_CLASSES> <SHOTS> [MODEL_CONFIG]

GPU=$1                                    # GPU ID
DATASET=$2                                # Dataset name
SUBSAMPLE_CLASSES=${3:-"base"}            # Class subset to test
SHOTS=${4:-8}                             # Number of shots
MODEL_CONFIG=${5:-"0513-latest-4"}        # Config for loading model weights

DATA="${COOP_DATASET}"                    # Dataset root path
cfg_file="fedmgp_vis_test"                # Visualization config
trainer=FedMGP
model=FedMGP
SEED=0
USEALL=False

if [ -z "$GPU" ] || [ -z "$DATASET" ]; then
  echo "Usage: $0 <GPU> <DATASET> [SUBSAMPLE_CLASSES] [SHOTS] [MODEL_CONFIG]"
  echo "Example: $0 0 caltech101 base 8 0513-latest-4"
  exit 1
fi

# Base model path (using MODEL_CONFIG)
BASE_DIR=output/${DATASET}/${trainer}/${MODEL_CONFIG}/${SHOTS}shots/seed${SEED}/base

# Test output path (using visualization config)
OUTPUT_DIR=output/${DATASET}/${trainer}/${cfg_file}/${SHOTS}shots/seed${SEED}/${SUBSAMPLE_CLASSES}_test_vis

if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Base model not found at ${BASE_DIR}"
  echo "Please train model first or check MODEL_CONFIG: ${MODEL_CONFIG}"
  exit 1
fi

# Find model weights
MODEL_PATH=""
if [ -f "${BASE_DIR}/best_model.pt" ]; then
  MODEL_PATH="${BASE_DIR}/best_model.pt"
  echo "Using best model: ${MODEL_PATH}"
elif [ -f "${BASE_DIR}/last_model.pt" ]; then
  MODEL_PATH="${BASE_DIR}/last_model.pt"
  echo "Using last model: ${MODEL_PATH}"
else
  echo "Error: Model weights not found"
  echo "Please check path: ${BASE_DIR}"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "FedMGP Attention Visualization Test"
echo "=========================================="
echo "GPU: ${GPU}"
echo "Dataset: ${DATASET}"
echo "Classes: ${SUBSAMPLE_CLASSES}"
echo "Shots: ${SHOTS}"
echo "Model config: ${MODEL_CONFIG}"
echo "Visualization config: ${cfg_file}"
echo "Model weights: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

echo "Testing on ${SUBSAMPLE_CLASSES} classes with visualization..."
CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
--root ${DATA} \
--output-dir ${OUTPUT_DIR} \
--seed ${SEED} \
--model ${model} \
--trainer ${trainer} \
--config-file configs/trainers/${trainer}/${cfg_file}.yaml \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--num_shots ${SHOTS} \
--eval-only \
--model-dir ${MODEL_PATH} \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.USEALL ${USEALL}

echo "=========================================="
echo "Visualization test completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Visualization plots saved to: ${OUTPUT_DIR}/fedmgp_attention_vis/"
echo "=========================================="
