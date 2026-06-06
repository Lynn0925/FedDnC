#!/bin/bash
# LODO on Office-Caltech10 with FedPGP
# Usage: bash scripts/FedPGP/office_lodo.sh <GPU> [SEED]

GPU=$1
SEED=${2:-0}

DATASET="office"
DATA="${COOP_DATASET}"
CFG="office_lodo_vit_b16"
TRAINER="FedPGP"
MODEL="FedPGP"

DOMAINS=("amazon" "caltech" "dslr" "webcam")

for TARGET in "${DOMAINS[@]}"; do
    DIR="output/${DATASET}/${TRAINER}/${CFG}/target_${TARGET}/seed${SEED}"
    mkdir -p "${DIR}"

    echo "============================================"
    echo "[FedPGP] Target domain: ${TARGET}"
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
        DATASET.TARGET_DOMAIN "${TARGET}" \
        DATASET.SUBSAMPLE_CLASSES "all" \
        DATASET.USEALL True \
        DATASET.USERS 3

    echo "Done: target=${TARGET}"
done

echo "All FedPGP Office-Caltech10 LODO runs completed."
