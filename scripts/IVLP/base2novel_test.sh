#!/bin/bash

GPU=$1                                    # GPU ID
DATASET=$2                                # Dataset name
SUBSAMPLE_CLASSES=${3:-"base"}            # Class subset to test (base/new)
SHOTS=${4:-8}                             # Number of shots, default 8
MODEL_DIR=${5:-""}                        # Path to trained model weights (.pt file)
cfg_file=base2novel_vit_b16

DATA="${COOP_DATASET}"                    # Dataset root path (set via env var)
trainer=IVLP
model=IVLP
SEED=0
USEALL=False

# Test output path
OUTPUT_DIR=output/${DATASET}/${trainer}/${cfg_file}/${SHOTS}shots/seed${SEED}/${SUBSAMPLE_CLASSES}_test

if [ -z "$MODEL_DIR" ]; then
  echo "Error: MODEL_DIR not specified. Usage: bash $0 <GPU> <DATASET> <base|new> <SHOTS> <path/to/model.pt>"
  exit 1
fi

if [ ! -f "$MODEL_DIR" ]; then
  echo "Error: Model weights not found at ${MODEL_DIR}"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

echo "Testing on ${SUBSAMPLE_CLASSES} classes..."
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
--model-dir ${MODEL_DIR} \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.USEALL ${USEALL}

echo "Testing completed, results saved to ${OUTPUT_DIR}"
