# Unified Replication of Backdoor Attacks and Defenses in Deep Learning

This repository is a **combined submission repository** for four research papers that were replicated as part of my work on backdoor attacks and defenses in deep learning.

Each paper was **originally implemented and evaluated in its own separate GitHub repository**.  
Those individual repositories are still available and can be referred to for the complete development and experiment history.

## Original Individual Repositories

- **Attack 1 — Narcissus (CCS 2023)**  
  https://github.com/abhinavbibek/deakin_research_attack1

- **Attack 2 — COMBAT (AAAI 2024)**  
  https://github.com/abhinavbibek/deakin_research_attack2

- **Defense 1 — Incompatibility Clustering (ISPL+B, ICLR 2023)**  
  https://github.com/abhinavbibek/deakin_research_defense1

- **Defense 2 — ASD: Adaptive Dataset Splitting (CVPR 2023)**  
  https://github.com/abhinavbibek/deakin_research_defense2

This master repository only **brings the finalized code from all four works into a single place** to make review and submission easier.

## Repository Structure

```text
ml-backdoor-attacks-and-defenses/
├── attack1/      # Narcissus: clean-label backdoor attack (CCS 2023)
├── attack2/      # COMBAT: alternated training backdoor attack (AAAI 2024)
├── defense1/     # Incompatibility Clustering (ISPL+B) defense (ICLR 2023)
├── defense2/     # ASD defense: adaptive dataset splitting (CVPR 2023)
└── README.md

```
## Part I — Backdoor Attack Replications

### Attack 1: Narcissus — CCS 2023

**Paper**: *Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information*  
**Conference**: ACM SIGSAC Conference on Computer and Communications Security (CCS 2023)

**Key Properties**
- Clean-label backdoor attack
- Extremely low poison rates (≤ 0.05%)
- All-to-one attack
- Requires only target-class data
- Effective on models trained from scratch
- Supports physical-world settings

**Datasets**
- CIFAR-10
- Tiny-ImageNet (Domain-Adapted ImageNet Surrogate)

**Location**
```
attack1/
```


### Attack 2: COMBAT — AAAI 2024

**Paper**: *COMBAT: Alternated Training for Effective Clean-Label Backdoor Attacks*  
**Conference**: AAAI Conference on Artificial Intelligence (AAAI 2024)

**Key Properties**
- Generator–surrogate alternated training framework
- Flexible trigger synthesis
- High attack success rate while preserving clean accuracy
- Demonstrated robustness against existing defenses

**Datasets**
- CIFAR-10
- CelebA

**Location**
```
attack2/
```


## Part II — Backdoor Defense Replications

### Defense 1: Incompatibility Clustering (ISPL+B) — ICLR 2023

**Paper**: *Incompatibility Clustering as a Defense Against Backdoor Poisoning Attacks*  
**Conference**: International Conference on Learning Representations (ICLR 2023)

**Threat Model**
- Dirty Label Backdoor (DLBD)
- 1-to-1 source–target attacks

**Core Idea**
- Poisoned samples are incompatible with clean data
- Inverse Self-Paced Learning (ISPL) isolates homogeneous clusters
- Boosting identifies the clean cluster
- Retraining on sanitized data removes backdoor behavior

**Datasets**
- CIFAR-10
- GTSRB

**Location**
```
defense1/
```


### Defense 2: ASD — CVPR 2023

**Paper**: *Backdoor Defense via Adaptively Splitting Poisoned Dataset*  
**Conference**: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023)

**Threat Model**
- BadNets (Many-to-One)
- Fixed visible trigger
- 5% poisoning rate

**Core Idea**
- Loss-guided dataset splitting using GMM
- Meta-learning-inspired refinement
- Adaptive retraining on clean data

**Datasets**
- CIFAR-10
- GTSRB

**Location**
```
defense2/
```

