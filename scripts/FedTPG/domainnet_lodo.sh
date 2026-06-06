#!/bin/bash
# LODO evaluation on DomainNet with FedPGP
# Usage: bash scripts/FedPGP/domainnet_lodo.sh <GPU> [SEED]

GPU=$1
SEED=${2:-0}

DATASET="domainnet"
DATA="${COOP_DATASET}"
CFG="domainnet_lodo_vit_b16"
TRAINER="FedTPG"
MODEL="FedTPG"

DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")

for TARGET in "${DOMAINS[@]}"; do
    DIR="output/${DATASET}/${TRAINER}/${CFG}/target_${TARGET}/seed${SEED}"
    mkdir -p "${DIR}"

    echo "============================================"
    echo "[FedTPG] Target domain: ${TARGET}"
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
        DATASET.USERS 5

    echo "Done: target=${TARGET}"
done

echo "All FedPGP LODO runs completed."
