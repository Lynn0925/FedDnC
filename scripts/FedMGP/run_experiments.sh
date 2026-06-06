#!/bin/bash

GPU=$1
SEED=${2:-0}

if [ -z "$GPU" ]; then
    echo "Error: GPU ID required as first argument"
    exit 1
fi

LOG_DIR=logs
mkdir -p ${LOG_DIR}

echo "Starting FedMGP experiments (GPU=${GPU}, SEED=${SEED})"
echo "====================================="

# Train DomainNet
echo "[1/4] Training DomainNet..."
DOMAINNET_OUTPUT_DIR=$(bash scripts/FedMGP/domainnet_train.sh ${GPU} ${SEED} | grep "Output:" | cut -d' ' -f2)
echo "DomainNet training completed, model saved to: ${DOMAINNET_OUTPUT_DIR}"
echo "====================================="

# Test DomainNet
echo "[2/4] Testing DomainNet..."
bash scripts/FedMGP/domainnet_test.sh ${GPU} ${SEED} ${DOMAINNET_OUTPUT_DIR} > ${LOG_DIR}/domainnet_test_results.log
echo "DomainNet testing completed, results saved to: ${LOG_DIR}/domainnet_test_results.log"
echo "====================================="

# Train Office
echo "[3/4] Training Office..."
OFFICE_OUTPUT_DIR=$(bash scripts/FedMGP/office_train.sh ${GPU} ${SEED} | grep "Output:" | cut -d' ' -f2)
echo "Office training completed, model saved to: ${OFFICE_OUTPUT_DIR}"
echo "====================================="

# Test Office
echo "[4/4] Testing Office..."
bash scripts/FedMGP/office_test.sh ${GPU} ${SEED} ${OFFICE_OUTPUT_DIR} > ${LOG_DIR}/office_test_results.log
echo "Office testing completed, results saved to: ${LOG_DIR}/office_test_results.log"
echo "====================================="

echo "All experiments completed!"
echo "Results summary:"
echo "DomainNet: ${LOG_DIR}/domainnet_test_results.log"
echo "Office: ${LOG_DIR}/office_test_results.log"

echo "DomainNet domain accuracy:"
grep "domain.*accuracy" ${LOG_DIR}/domainnet_test_results.log

echo "Office domain accuracy:"
grep "domain.*accuracy" ${LOG_DIR}/office_test_results.log

echo "Experiments finished"
