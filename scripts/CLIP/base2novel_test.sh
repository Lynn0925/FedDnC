#!/bin/bash

GPU=$1                                    # GPU ID
DATASET=$2                                # Dataset name
SUBSAMPLE_CLASSES=${3:-"base"}            # Class subset to test (base/new)
SHOTS=${4:-16}                            # Number of shots, default 16
CONFIG_FILE=${5:-"base2novel_vit_b16"}    # Config file name

DATA="${COOP_DATASET}"                    # Dataset root path
cfg_file=${CONFIG_FILE}
trainer=CLIP
model=CLIP
SEED=0
USEALL=False

# Test output path
OUTPUT_DIR=output/${DATASET}/${trainer}/${cfg_file}/${SHOTS}shots/seed${SEED}/${SUBSAMPLE_CLASSES}_test

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
--subsample ${SUBSAMPLE_CLASSES} \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.USEALL ${USEALL}

echo "Testing completed, results saved to ${OUTPUT_DIR}"
