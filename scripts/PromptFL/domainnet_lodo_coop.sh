#!/bin/bash
# LODO evaluation on DomainNet with local (non-federated) CoOp
# Each client trains its own prompt independently — no weight sharing.
# Usage: bash scripts/PromptFL/domainnet_lodo_coop.sh <GPU> [SEED]

GPU=$1
SEED=${2:-0}

DATASET="domainnet"
DATA="${COOP_DATASET}"
CFG="domainnet_lodo_vit_b16"   # reuse PromptFL LODO YAML (same architecture)
TRAINER="PromptFL"
MODEL="LocalCoOp"               # non-federated per-client learner

DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")

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
        DATASET.USERS 5

    echo "Done: target=${TARGET}"
done

echo "All CoOp (local) LODO runs completed."
