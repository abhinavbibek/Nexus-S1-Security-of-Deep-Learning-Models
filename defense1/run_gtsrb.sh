#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Running GTSRB DLBD on GPU ${CUDA_VISIBLE_DEVICES}"

for i in {0..23}; do
  echo "GTSRB scenario $i"
  python -m evaluation.backdoor_tests_gtsrb $i preactresnet18 \
    | tee logs/gtsrb/run_$i.txt
done

echo "GTSRB experiments completed."
