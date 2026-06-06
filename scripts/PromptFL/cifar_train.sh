#!/bin/bash

GPU=$1                    # GPU ID
DATASET=$2                # cifar10 or cifar100
SEED=${3:-0}              # Random seed, default 0

if [[ "$DATASET" != "cifar10" && "$DATASET" != "cifar100" ]]; then
    echo "Error: Dataset must be 'cifar10' or 'cifar100'"
    exit 1
fi

DATA="${COOP_DATASET}"    # Dataset root path
cfg_file="cifar_train"
trainer=PromptFL
model=PromptFL
SUBSAMPLE_CLASSES="local"
USEALL=True

DIR=output/${DATASET}/${trainer}/${cfg_file}/seed${SEED}

mkdir -p ${DIR}

echo "Training ${DATASET}..."
echo "Config: GPU=${GPU}, Dataset=${DATASET}, Seed=${SEED}"
echo "Params: 100 rounds, 100 clients, 10% participation, alpha=0.5"
echo "Output: ${DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
  --root ${DATA} \
  --output-dir ${DIR} \
  --seed ${SEED} \
  --model ${model} \
  --trainer ${trainer} \
  --config-file configs/trainers/${trainer}/${cfg_file}.yaml \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
  DATASET.USEALL ${USEALL}

echo "Training completed! Results saved to ${DIR}"
