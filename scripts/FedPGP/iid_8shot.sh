#!/bin/bash
# IID 8-shot experiments for FedPGP
# Usage: bash scripts/FedPGP/iid_8shot.sh <GPU> <DATASET> [SEED] [SHOTS]
# Example: bash scripts/FedPGP/iid_8shot.sh 0 caltech101 0 8

GPU=$1
DATASET=$2
SEED=${3:-0}
SHOTS=${4:-8}

DATA="${COOP_DATASET}"
CFG="base2novel_vit_b16"
TRAINER="FedPGP"
MODEL="FedPGP"

DIR="output/${DATASET}/${TRAINER}/iid_${SHOTS}shots/seed${SEED}/all"
mkdir -p "${DIR}"

echo "============================================"
echo "[FedPGP] IID ${SHOTS}-shot on ${DATASET}"
echo "Output: ${DIR}"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
    --root "${DATA}" \
    --output-dir "${DIR}" \
    --seed ${SEED} \
    --model ${MODEL} \
    --trainer ${TRAINER} \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --num_shots ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES "all" \
    DATASET.USEALL False \
    DATASET.IID True

echo "Done: FedPGP IID ${SHOTS}-shot on ${DATASET}"
