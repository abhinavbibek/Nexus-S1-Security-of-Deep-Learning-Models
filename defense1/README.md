# Incompatibility Clustering Defense (ISPL+B)

This repository contains an implementation of the defense mechanism described in the ICLR 2023 paper: **Incompatibility Clustering as a Defense Against Backdoor Poisoning Attacks**.

## Paper Details

*   **Title**: Incompatibility Clustering as a Defense Against Backdoor Poisoning Attacks
*   **Authors**: Charles Jin, Melinda Sun, Martin Rinard (MIT)
*   **Conference**: [ICLR 2023](https://iclr.cc/virtual/2023/poster/11486)
*   **Paper Link**: [OpenReview / arXiv](https://arxiv.org/abs/2105.03692)

**Citation**:
```bibtex
@inproceedings{jin2023incompatibility,
  title={Incompatibility Clustering as a Defense Against Backdoor Poisoning Attacks},
  author={Jin, Charles and Sun, Melinda and Rinard, Martin},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## Threat Model: Dirty Label Backdoor (DLBD) Attack

This implementation focuses on defending against the **Dirty Label Backdoor (DLBD)** attack.

*   **Objective**: The attacker aims to install a backdoor that causes the model to misclassify specific inputs (containing a "trigger") as a target class, while maintaining high accuracy on clean data.
*   **Mechanism**: The attacker injects a small set of poisoned images into the training set. These images have a visible trigger (e.g., a pixel patch) and are mislabeled as the target class.
*   **Scenario (1-to-1)**:
    *   **Source Class**: The class the attacker wants to misclassify (e.g., "Airport").
    *   **Target Class**: The class the attacker wants the model to predict (e.g., "Bird").
    *   **Trigger**: A specific pattern added to the source class images.
    *   **Poison Rate ($\epsilon$)**: The percentage of the source class that is poisoned (e.g., 5%, 10%, 20%).

## Defense Mechanism: Incompatibility Clustering (ISPL+B)

Our defense leverages the insight that **poisoned data is "incompatible" with clean data**. That is, a model trained on clean data does not generalize well to poisoned data (and vice-versa).

### 1. Incompatibility Property
We define two sets of data as **incompatible** if training on one does not improve the model's performance on the other. This property allows us to separate the dataset into homogeneous clusters (e.g., one cluster of mostly clean data, one of mostly poisoned data).

### 2. Inverse Self-Paced Learning (ISPL)
This algorithm iteratively identifies and separates the "most compatible" subset of data.
*   It trains a model on a subset of data.
*   It selects samples with the **lowest loss** (assuming they are the most "compatible" or consistent with the current model).
*   By annealing the subset size, it converges to a highly homogeneous cluster (often isolating the poisoned data, as they are strongly correlated with the trigger).

### 3. Boosting (+B)
Once ISPL partitions the data into clusters, we need to identify which cluster is "clean" and which is "poisoned".
*   We use a **Voting/Boosting** mechanism.
*   We train a weak learner on each cluster and have them vote on the remaining data.
*   Since the clean distribution is dominant and self-compatible, the majority vote correctly identifies the clean set.

### 4. Sanitization & Retraining
The final step is to **remove the identified poisoned cluster** from the training set and retrain the victim model (ResNet) from scratch on the sanitized dataset.

## Implementation & Reproduction

We provide a complete pipeline to reproduce the results for CIFAR-10 and GTSRB.

### 1. Installation

We used Python 3.8.1 for all experiments. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Generation

First, generate the poisoned datasets. This script creates the DLBD datasets with the specific triggers and source-target pairs used in our experiments.

```bash
mkdir -p datasets
python -m data.cifar10_backdoor
# The script generates datasets for different random seeds and epsilon values.
```

### 3. Running the Defense

We have provided shell scripts to automate the evaluation across all 24 attack scenarios (0-23) for each dataset.

**For CIFAR-10:**
```bash
./run_cifar10.sh
```

**For GTSRB:**
```bash
./run_gtsrb.sh
```

These scripts execute `evaluation.backdoor_tests`, which performs the following for each scenario:
1.  Loads the poisoned dataset.
2.  Runs ISPL to cluster the data.
3.  Runs Boosting to identify the clean set.
4.  Retrains the model on the sanitized set.
5.  Evaluates the final model on Clean Test Data and Poisoned Test Data.

### 4. Evaluation & Results

After execution, you can parse the logs to aggregate the metrics.

**Parse Logs:**
```bash
python results/parse_logs.py
```
This generates `dataset/all_metrics.csv`.

**Calculate Summary Statistics:**
```bash
python results/calculate_detailed_stats.py
```
This script outputs the **Mean Attack Success Rate (ASR)** and **Mean Clean Accuracy** with standard deviations.

**Success Rate Comparison (Table 1):**
```bash
python results/table1.py
```
This allows you to compare the "Success/Total" count against the paper's Table 1.

## Results Comparison

Below is a comparison of the defense success rates (Epsilon vs Success Count/Total runs) between our reproduction and the original paper's results for the DLBD 1-to-1 setting.

### CIFAR-10 (DLBD 1-to-1)

| Epsilon | Our Results | Paper Results |
| :--- | :--- | :--- |
| **0.05** | **7 / 8** | 7 / 8 |
| **0.1** | **8 / 8** | 8 / 8 |
| **0.2** | **7 / 8** | 7 / 8 |
| **Total** | **22 / 24** | **22 / 24** |

### GTSRB (DLBD 1-to-1)

| Epsilon | Our Results | Paper Results |
| :--- | :--- | :--- |
| **0.05** | **7 / 8** | 6 / 8 |
| **0.1** | **6 / 8** | 6 / 8 |
| **0.2** | **4 / 8** | 5 / 8 |
| **Total** | **17 / 24** | **17 / 24** |

*Note: An entry of "7 / 8" indicates that ISPL+B successfully defended 7 of the 8 scenarios in that setting (Success defined as TMR < 1%).*
