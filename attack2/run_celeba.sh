#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

python train_clean_classifier.py --dataset celeba --saving_prefix classifier_clean_celeba

# Paper Setting: Overall Poisoning Rate (p) = 5% (0.05)
# Calculation: pc (Target Class Poisoning) = p * NumClasses
# For CelebA (8 Classes): pc = 0.05 * 8 = 0.4

python train_generator_celeba.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_generator_celeba_pc04_n00392 \
  --load_checkpoint_clean classifier_clean_celeba


python train_victim_celeba.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_celeba_pc04_n00392 \
  --load_checkpoint train_generator_celeba_pc04_n00392_clean


python eval_celeba.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_celeba_pc04_n00392 \
  --load_checkpoint_clean classifier_clean_celeba \
  --load_checkpoint train_generator_celeba_pc04_n00392_clean
