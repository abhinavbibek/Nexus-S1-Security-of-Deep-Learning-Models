# COMBAT Attack Replication

In this project, we replicate the implementation of the paper **“COMBAT: Alternated Training for Effective Clean-Label Backdoor Attacks”**, which was presented at the **AAAI Conference on Artificial Intelligence (AAAI 2024)**. We successfully reproduce the attack on two datasets: **CIFAR-10** and **CelebA**.

Our experimental results closely match—and in some cases slightly exceed—the results reported in the original paper, confirming the reproducibility and effectiveness of the proposed alternated training framework.

**Paper Link**: [COMBAT: Alternated Training for Effective Clean-Label Backdoor Attacks](https://ojs.aaai.org/index.php/AAAI/article/view/28019)

**Citation**:
> Huynh, T., Nguyen, D., Pham, T., & Tran, A. (2024).  
> COMBAT: Alternated Training for Effective Clean-Label Backdoor Attacks.  
> In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*,  
> Vol. 38, No. 3, pp. 2436–2444.

---

## COMBAT: Alternated Training for Effective Clean-Label Backdoor Attack

COMBAT is a novel mechanism for creating highly effective clean-label attacks using a trigger pattern generator trained alongside a surrogate model. This flexible approach allows for various backdoor trigger types and targets, achieving near-perfect attack success rates and evading all advanced backdoor defenses, as demonstrated through extensive experiments on standard datasets (CIFAR-10, CelebA, ImageNet-10).

---


## Requirements
Install required Python packages:
```
$ python -m pip install -r requirements.txt
```
## Training clean model
Run command
```
$ python train_clean_classifier.py --dataset <datasetName> --saving_prefix <cleanModelPrefix>
```

where the parameters are as following:
- `dataset`: name of the dataset used for training (`cifar10` | `imagenet10` | `celeba`)
- `saving_prefix`: prefix for saving the trained clean model checkpoint
  
The trained checkpoint of the clean model should be saved at the path `checkpoints\<cleanModelPrefix>\<datasetName>\<datasetName>_<cleanModelPrefix>.pth.tar.`


## Training trigger generator and surrogate model 
Run command
```
$ python train_generator.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint_clean <cleanModelPrefix>
``` 

where the parameters are as following:
- `dataset`: name of the dataset used for training (`cifar10` | `imagenet10` | `celeba`)
- `pc`: proportion of the target class data to poison on a 0-to-1 scale
- `noise_rate`: strength/amplitude of the backdoor trigger on a 0-to-1 scale
- `saving_prefix`: prefix for saving the trained generator and surrogate model checkpoint
- `load_checkpoint_clean`: prefix of the trained clean model checkpoint

The trained checkpoint of the generator and surrogate model should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`

## Train victim model
Run command
```
$ python train_victim.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint <trainedCheckpoint>
```
- `dataset`: name of the dataset used for training (`cifar10` | `imagenet10` | `celeba`)
- `pc`: proportion of the target class data to poison on a 0-to-1 scale
- `noise_rate`: strength/amplitude of the backdoor trigger on a 0-to-1 scale
- `saving_prefix`: prefix for saving the trained victim model checkpoint
- `load_checkpoint`: trained generator checkpoint folder name

The trained checkpoint of the victim model should be saved at the path `checkpoints\<savingPrefix>_clean\<datasetName>\<datasetName>_<savingPrefix>_clean.pth.tar.`

## Evaluate victim model
Run command
```
$ python eval.py --dataset <datasetName> --pc <poisoningRate> --noise_rate <triggerStrength> --saving_prefix <savingPrefix> --load_checkpoint_clean <cleanModelPrefix> --load_checkpoint <trainedCheckpoint>
```
- `dataset`: name of the dataset used for training (`cifar10` | `imagenet10` | `celeba`)
- `pc`: proportion of the target class data to poison on a 0-to-1 scale
- `noise_rate`: strength/amplitude of the backdoor trigger on a 0-to-1 scale
- `saving_prefix`: prefix for saving the trained victim model checkpoint
- `load_checkpoint_clean`: trained clean model checkpoint folder name
- `load_checkpoint`: trained generator checkpoint folder name
  
## Sample Runs

### 1. CIFAR-10 Dataset
```bash
# 1. Train Clean Classifier
python train_clean_classifier.py --dataset cifar10 --saving_prefix classifier_clean

# 2. Train Generator (Poisoning Rate: 0.5, Trigger Strength: 0.0392)
python train_generator.py \
  --dataset cifar10 \
  --pc 0.5 \
  --noise_rate 0.0392 \
  --saving_prefix train_generator_pc05_n00392_tuned \
  --load_checkpoint_clean classifier_clean

# 3. Train Victim Model with Backdoor
python train_victim.py \
  --dataset cifar10 \
  --pc 0.5 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_pc05_n00392_tuned \
  --load_checkpoint train_generator_pc05_n00392_tuned_clean

# 4. Evaluate Attack Success
python eval.py \
  --dataset cifar10 \
  --pc 0.5 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_pc05_n00392_tuned \
  --load_checkpoint_clean classifier_clean \
  --load_checkpoint train_generator_pc05_n00392_tuned_clean
```

### 2. CelebA Dataset
```bash
# 1. Train Clean Classifier
python train_clean_classifier.py --dataset celeba --saving_prefix classifier_clean_celeba

# 2. Train Generator (Poisoning Rate: 0.4, Trigger Strength: 0.0392)
python train_generator.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_generator_celeba_pc04_n00392 \
  --load_checkpoint_clean classifier_clean_celeba

# 3. Train Victim Model with Backdoor
python train_victim.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_celeba_pc04_n00392 \
  --load_checkpoint train_generator_celeba_pc04_n00392_clean

# 4. Evaluate Attack Success
python eval.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_celeba_pc04_n00392 \
  --load_checkpoint_clean classifier_clean_celeba \
  --load_checkpoint train_generator_celeba_pc04_n00392_clean
```


## Results
We successfully replicated the attack on **CIFAR-10** and **CelebA** datasets. Below is a detailed comparison between our results and the values reported in the original paper (Table 1).

### 1. CIFAR-10
| Metric | Paper Reported | Our Result | Difference |
| :--- | :---: | :---: | :---: |
| **Attack Success Rate (ASR)** | 97.73% | **97.99%** | **+0.26%** |
| **Clean Accuracy (BA)** | 94.58% | 94.39% | -0.19% |


### 2. CelebA
| Metric | Paper Reported | Our Result | Difference |
| :--- | :--- | :---: | :---: | 
| **Attack Success Rate (ASR)** | 99.84% | **99.87%** | **+0.03%** |
| **Clean Accuracy (BA)** | 79.34% | 79.28% | -0.06% |



*Note: Clean Accuracy (BA) refers to the model's performance on clean data, while Attack Success Rate (ASR) measures the success of the backdoor on the target class.*
