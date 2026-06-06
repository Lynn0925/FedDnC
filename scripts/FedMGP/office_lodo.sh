#!/bin/bash
# Leave-One-Domain-Out evaluation on Office-Caltech10 (following FedPGP paper setup)
# Usage: bash scripts/FedDnC/office_lodo.sh <GPU> [SEED]
# Example: bash scripts/FedDnC/office_lodo.sh 0 0

GPU=$1
SEED=${2:-0}

DATASET="office"
DATA="${COOP_DATASET}"
CFG="office_lodo_vit_b16"
TRAINER="FedMGP"

DOMAINS=("amazon" "caltech" "dslr" "webcam")

for TARGET in "${DOMAINS[@]}"; do
    DIR="output/${DATASET}/${TRAINER}/${CFG}/target_${TARGET}/seed${SEED}"
    mkdir -p "${DIR}"

    echo "============================================"
    echo "Target domain: ${TARGET}"
    echo "Output: ${DIR}"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
        --root "${DATA}" \
        --output-dir "${DIR}" \
        --seed ${SEED} \
        --model ${TRAINER} \
        --trainer ${TRAINER} \
        --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        DATASET.TARGET_DOMAIN "${TARGET}" \
        DATASET.SUBSAMPLE_CLASSES "all" \
        DATASET.USEALL True \
        DATASET.USERS 3

    echo "Done: target=${TARGET}"
done

echo "============================================"
echo "All LODO runs completed."
echo "Results in: output/${DATASET}/${TRAINER}/${CFG}/"
echo "============================================"
