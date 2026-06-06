#!/bin/bash

# Usage: ./test_debug.sh <GPU> <DATASET> <SUBSAMPLE_CLASSES> <SHOTS> [TRAIN_CONFIG] [TEST_CONFIG]

GPU=$1                                      # GPU ID
DATASET=$2                                  # Dataset name
SUBSAMPLE_CLASSES=${3:-"base"}              # Class subset to test
SHOTS=${4:-16}                              # Number of shots
TRAIN_CONFIG_FILE=${5:-"0513-latest-4"}     # Training config (for weight path)
TEST_CONFIG_FILE=${6:-"$TRAIN_CONFIG_FILE"} # Testing config

DATA="${COOP_DATASET}"                      # Dataset root path
trainer=FedMGP
model=FedMGP
SEED=0
USEALL=False

# Base model path (based on training config)
BASE_DIR=output/${DATASET}/${trainer}/${TRAIN_CONFIG_FILE}/${SHOTS}shots/seed${SEED}/base

# Test output path (based on test config)
OUTPUT_DIR=output/debug/${DATASET}/${trainer}/${TEST_CONFIG_FILE}/${SUBSAMPLE_CLASSES}_test

if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Base model not found at ${BASE_DIR}"
  echo "Please check training config: ${TRAIN_CONFIG_FILE}"
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
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

echo "==================================================================================="
echo "Testing on ${SUBSAMPLE_CLASSES} classes..."
echo "Training config: ${TRAIN_CONFIG_FILE} (weight path)"
echo "Testing config: ${TEST_CONFIG_FILE} (test params)"
echo "Model weights: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "==================================================================================="

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
--root ${DATA} \
--output-dir ${OUTPUT_DIR} \
--seed ${SEED} \
--model ${model} \
--trainer ${trainer} \
--config-file configs/trainers/${trainer}/${TEST_CONFIG_FILE}.yaml \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--num_shots ${SHOTS} \
--eval-only \
--model-dir ${MODEL_PATH} \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.USEALL ${USEALL}

echo "==================================================================================="
echo "Testing completed, results saved to ${OUTPUT_DIR}"
echo "Similarity plots saved to ${OUTPUT_DIR}/similarity_plots/ (if visualization enabled)"
echo "==================================================================================="
