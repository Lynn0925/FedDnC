#!/bin/bash
# LODO on Office-Caltech10 with local (non-federated) CoOp
# Usage: bash scripts/PromptFL/office_lodo_coop.sh <GPU> [SEED]

GPU=$1
SEED=${2:-0}

DATASET="office"
DATA="${COOP_DATASET}"
CFG="office_lodo_vit_b16"
TRAINER="PromptFL"
MODEL="LocalCoOp"

DOMAINS=("amazon" "caltech" "dslr" "webcam")

for TARGET in "${DOMAINS[@]}"; do
    DIR="output/${DATASET}/CoOp/${CFG}/target_${TARGET}/seed${SEED}"
    mkdir -p "${DIR}"

    echo "============================================"
    echo "[CoOp local] Target domain: ${TARGET}"
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

echo "All CoOp (local) Office-Caltech10 LODO runs completed."
