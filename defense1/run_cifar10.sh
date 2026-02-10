#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Running CIFAR-10 DLBD on GPU ${CUDA_VISIBLE_DEVICES}"

for i in {0..23}; do
  echo "CIFAR-10 scenario $i"
  python -m evaluation.backdoor_tests $i preactresnet18 \
    | tee logs/cifar10/run_$i.txt
done

echo "CIFAR-10 experiments completed."
