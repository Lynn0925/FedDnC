# FedDnC: Divide and Conquer — Federated Prompt Learning via Dual-Stream Prompts for Vision-Language Models

> **FedDnC** (*Federated Divide-and-Conquer*) resolves the fundamental conflict in federated prompt learning between discriminative adaptation and zero-shot generalization by maintaining two *architecturally decoupled* prompt streams. It achieves **82.29% average HM** across seven recognition benchmarks — the highest among all evaluated federated methods — while transmitting only **8.0 KB per round**, the smallest communication footprint.

---

## Abstract

Federated Prompt Learning (FPL) enables parameter-efficient adaptation of vision-language models across distributed clients without sharing raw data. However, under non-IID distributions a single shared prompt cannot simultaneously sustain discriminative adaptation and zero-shot generalization: local cross-entropy training drives prompts toward client-specific decision boundaries, while generalization requires preserving the broad semantic coverage of the pretrained CLIP space. These two objectives are structurally incompatible.

**FedDnC** resolves this by *architectural decoupling*. It maintains two independent learnable prompt streams:

- **Discriminative stream P_b** — randomly initialized, trained with cross-entropy + FedProx, aggregated via data-weighted FedAvg to consolidate cross-client discriminative consensus.
- **Generalization stream P_n** — initialized from the zero-shot template `"a photo of a"`, trained with KL-divergence and cosine alignment to the pretrained CLIP space, aggregated via EMA to resist non-IID noise and maintain stable zero-shot alignment.

At inference, a per-sample **confidence-adaptive fusion** weight α_i dynamically interpolates between the two streams without any held-out validation data.

---

## Key Results

### Base-to-Novel Generalization (8-shot, K=10 clients, β=0.3)

| Method | Flowers | DTD | Caltech | UCF101 | Food101 | Cars | Pets | **Avg HM** |
|--------|---------|-----|---------|--------|---------|------|------|-----------|
| CLIP (ZS)† | 72.88 | 56.37 | 95.66 | 71.38 | 92.38 | 68.89 | 92.94 | — |
| MaPLe† | 83.30 | 61.00 | 95.55 | 78.41 | 92.92 | 72.65 | 96.39 | — |
| PromptFL | 56.00 | 73.19 | 95.88 | 64.74 | 87.54 | 69.98 | 64.05 | 73.05 |
| FedTPG | 56.66 | **92.31** | **97.97** | 57.72 | 90.09 | 65.76 | 67.63 | 75.45 |
| FedOTP | 40.78 | 52.57 | 87.07 | 23.09 | 62.25 | 35.88 | 55.97 | 51.09 |
| FedPGP | 70.27 | 86.67 | 96.36 | 64.24 | 90.57 | 70.10 | 82.98 | 80.17 |
| **FedDnC (ours)** | **74.06** | 88.06 | **97.97** | **70.38** | **90.22** | **72.86** | **82.48** | **82.29** |

† Centralized upper-bound (trained on full pooled data), not directly comparable.

FedDnC improves novel-class accuracy over FedPGP (strongest baseline) by **+1.55% on average**, with particularly large margins on Flowers102 (+4.71%) and UCF101 (+5.24%).

### Domain Generalization (Leave-One-Domain-Out)

| Method | DomainNet Avg | Office-Caltech Avg |
|--------|--------------|-------------------|
| FedPGP | 86.12 | 95.88 |
| FedMGP | 85.26 | 96.57 |
| **FedDnC (ours)** | **88.44** | **96.84** |

### Communication Efficiency

| Method | Params | Upload/round | Time/round | GPU Mem | HM |
|--------|--------|-------------|-----------|---------|-----|
| PromptFL | 8.2K | 16.0 KB | 5.5 s | 2,061 MB | 56.00 |
| FedTPG | 4,206K | 8,215.0 KB | 6.7 s | 5,586 MB | 56.66 |
| FedOTP | 16.4K | 16.0 KB | 8.1 s | 6,748 MB | 40.78 |
| FedPGP | 24.8K | 32.0 KB | 16.1 s | 9,394 MB | 70.27 |
| **FedDnC (ours)** | **4.1K** | **8.0 KB** | 12.3 s | 9,852 MB | **74.06** |

FedDnC achieves the highest HM with the smallest communication footprint — 4× less upload than FedPGP and 1,027× less than FedTPG.

---

## Method

### Dual Prompt Design

```
For each class c and stream s ∈ {b, n}:

  t_c^{s,(i)} = [SOS], P_s^{(i,1)}, ..., P_s^{(i,L)}, e_c, [EOS]

  f_c^s = (1/N) Σ_i  ft(t_c^{s,(i)})          (averaged over N replicas)

P_b ∈ R^{N×L×d}  initialized ~ N(0, 0.02²)
P_n ∈ R^{N×L×d}  initialized from TokenEmb("a photo of a")
F_zs = {ft("a photo of a [class].")}_{c∈C}     (frozen throughout)
```

### Training Objective

```
L = L_CE
  + λ_kd · (L_kd + L_align)
  + β_div · (L_feat + L_div)
  + µ · L_prox

L_CE    = cross-entropy on logits ℓ_b = τ F_b v
L_kd    = T² · KL( σ(ℓ_b/T) ‖ σ(ℓ_zs/T) )       T=4
L_align = 1 - (1/|C|) Σ_c cos(f_c^n, f_c^zs)     anchors P_n to zero-shot
L_feat  = 1 - (1/|C|) Σ_c cos(f_c^b, f_c^zs)     prevents P_b from drifting
L_div   = (1/|C|) Σ_c cos(sg(f_c^b), f_c^n)       pushes P_n away from P_b
L_prox  = ‖P_b - P_b^(0)‖²_F                      FedProx on P_b
```

### Federated Aggregation

```
# P_b: data-weighted FedAvg (fast consensus)
P_b^{t+1} = Σ_{k∈S^t} (n_k / Σ_j n_j) · P_b,k

# P_n: EMA aggregation (slow, stable zero-shot alignment)
P_n^{t+1} = α · P_n^t + (1-α) · Σ_{k∈S^t} (n_k / Σ_j n_j) · P_n,k
             α = 0.9
```

### Confidence-Adaptive Inference

```
ℓ_b, ℓ_n = τ F_b fv(x),  τ F_n fv(x)

α_i = clamp( max σ(ℓ_b) / (max σ(ℓ_b) + max σ(ℓ_n)),  0.3,  0.9 )

ŷ = argmax( α_i · ℓ_b + (1-α_i) · ℓ_n )
```

When P_b is highly confident, α_i → 0.9; when both streams are uncertain, α_i → 0.5. No held-out validation data is required.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-username>/FedDnC.git
cd FedDnC

# 2. Create environment
conda create -n fednc python=3.9 -y
conda activate fednc

# 3. Install PyTorch (adjust CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

See [DATASETS.md](DATASETS.md) for full instructions. Set the root path:

```bash
export COOP_DATASET=/path/to/DATA
```

Expected layout:
```
DATA/
├── caltech-101/        ├── dtd/            ├── oxford_flowers/
├── cifar-10/           ├── eurosat/        ├── oxford_pets/
├── cifar-100/          ├── fgvc_aircraft/  ├── stanford_cars/
├── domainnet/          ├── food-101/       ├── sun397/
│                       ├── imagenet/       └── ucf101/
│                       └── office_caltech_10/
```

---

## Running Experiments

### Experiment 1: Base-to-Novel Generalization

Train on base classes, evaluate on base and novel (unseen) classes.

```bash
# Train (8-shot, 10 clients, 100 rounds)
bash scripts/FedDnC/base2novel_train.sh 0 flowers102 8
bash scripts/FedDnC/base2novel_train.sh 0 dtd 8
bash scripts/FedDnC/base2novel_train.sh 0 ucf101 8
# ... (caltech101, oxford_pets, stanford_cars, food101)

# Evaluate on base classes
bash scripts/FedDnC/base2novel_test.sh 0 flowers102 base 8

# Evaluate on novel (unseen) classes
bash scripts/FedDnC/base2novel_test.sh 0 flowers102 new 8
```

Supported datasets: `caltech101`, `oxford_pets`, `stanford_cars`, `oxford_flowers`, `food101`, `fgvc_aircraft`, `sun397`, `dtd`, `eurosat`, `ucf101`, `imagenet`

---

### Experiment 2: DomainNet — Non-IID Multi-Domain

30 clients, Dirichlet(α=0.3) label split, full data.

```bash
bash scripts/FedDnC/domainnet_train.sh 0 0   # GPU=0, Seed=0
bash scripts/FedDnC/domainnet_test.sh  0 0
```

---

### Experiment 3: Leave-One-Domain-Out Generalization

Each domain in turn is held out as the unseen test domain.

```bash
# DomainNet LODO (6 target domains: clipart, infograph, painting, quickdraw, real, sketch)
bash scripts/FedDnC/domainnet_lodo.sh 0 0

# Office-Caltech10 LODO (4 target domains: amazon, caltech, dslr, webcam)
bash scripts/FedDnC/office_lodo.sh 0 0
```

---

### Experiment 4: Office-Caltech10 Non-IID

```bash
bash scripts/FedDnC/office_train.sh 0 0
```

---

## Configuration

All hyperparameters are in `configs/trainers/FedDnC/`.

### Default Hyperparameters (from paper, `base2novel_vit_b16.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRAINER.FEDNC.N_CTX` | `4` | Context token length L |
| `TRAINER.FEDNC.N` | `2` | Prompt replicas per stream |
| `TRAINER.FEDNC.LAMBDA_KD` | `1.0` | Weight for L_kd + L_align |
| `TRAINER.FEDNC.BETA_DIV` | `0.1` | Weight for L_feat + L_div |
| `TRAINER.FEDNC.MU_PROX` | `0.01` | FedProx proximal weight µ |
| `TRAINER.FEDNC.EMA_ALPHA` | `0.9` | EMA momentum α for P_n |
| `TRAINER.FEDNC.KD_TEMP` | `4.0` | KD temperature T |
| `OPTIM.ROUND` | `100` | Communication rounds T |
| `OPTIM.MAX_EPOCH` | `5` | Local epochs per round E |
| `OPTIM.LR` | `0.002` | SGD learning rate |
| `DATASET.USERS` | `10` | Number of clients K |
| `DATASET.NUM_SHOTS` | `8` | Few-shot examples per class |
| `DATASET.BETA` | `0.3` | Dirichlet concentration β |

> Loss weights were tuned on Oxford Flowers102 and held **fixed** across all other datasets.

Command-line overrides:

```bash
python federated_main.py \
  --model FedDnC --trainer FedDnC \
  --config-file configs/trainers/FedDnC/base2novel_vit_b16.yaml \
  --dataset-config-file configs/datasets/ucf101.yaml \
  --num_shots 8 --seed 0 \
  TRAINER.FEDNC.LAMBDA_KD 1.0 \
  TRAINER.FEDNC.EMA_ALPHA 0.9 \
  DATASET.USERS 20
```

---

## Ablation Study Results

From Table V in the paper (UCF101 / DTD, β=0.3):

| Variant | UCF101 HM | DTD HM |
|---------|-----------|--------|
| w/o L_kd | 67.60 | 81.10 |
| w/o L_align | 64.52 | **90.33** |
| w/o L_feat | 65.59 | 85.37 |
| w/o L_div | 65.55 | 85.01 |
| P_b only (single stream) | 70.38 | 82.49 |
| FedAvg for P_n (no EMA) | 70.21 | 72.96 |
| Fixed α=0.5 (no adaptive) | 66.15 | 81.20 |
| **FedDnC (full)** | **70.38** | **88.06** |

Key findings:
- Replacing EMA with FedAvg for P_n causes **−15.10% HM** on DTD (texture domain), confirming that EMA is critical for preserving zero-shot alignment under non-IID noise.
- Removing L_div causes **−8.33% novel accuracy** on DTD, showing that without diversity loss the two streams silently collapse into equivalent representations.
- Adaptive α outperforms fixed α=0.5 by **+6.63% novel accuracy** on UCF101.

---

## Heterogeneity Robustness

From Table III (UCF101, varying β):

| Method | β=0.05 | β=0.1 | β=0.3 | β=0.5 | β=1.0 | Avg |
|--------|--------|-------|-------|-------|-------|-----|
| PromptFL | 64.25 | 59.94 | 65.65 | 67.76 | 64.93 | 64.51 |
| FedTPG | 64.03 | 68.32 | 63.79 | 72.15 | 70.89 | 67.84 |
| FedPGP | 67.09 | 71.43 | 67.07 | 66.36 | 64.62 | 67.31 |
| **FedDnC** | **73.82** | **71.43** | **71.94** | 67.17 | 65.49 | **69.97** |

FedDnC's advantage is most pronounced at β=0.05 (extreme non-IID): **+6.74% over FedPGP**, directly attributable to EMA's smoothing effect on round-to-round heterogeneous noise.

---

## Baseline Methods

All baselines are implemented and can be run with the same framework:

| Method | Key Idea | Config |
|--------|----------|--------|
| CLIP (ZS) | Zero-shot, no training | `configs/trainers/CLIP/` |
| PromptFL | FedAvg over single CoOp prompt | `configs/trainers/PromptFL/` |
| FedCoCoOp | Federated conditional context optimization | `configs/trainers/FedCoCoOp/` |
| IVLP | Independent vision-language prompts | `configs/trainers/IVLP/` |
| MaPLe | Multi-modal prompt learning | `configs/trainers/MaPLe/` |
| FedPGP | Prompt group personalization | `configs/trainers/FedPGP/` |
| FedOPT | Optimal transport prompt aggregation | `configs/trainers/FedOPT/` |
| FedTPG | Text-prompt guided generation | `configs/trainers/FedTPG/` |
| FedMGP | Multi-group text-visual prompts | `configs/trainers/FedMGP/` |

```bash
# Example: run FedPGP baseline
bash scripts/FedPGP/base2novel_train.sh 0 ucf101 8
bash scripts/FedPGP/base2novel_test.sh  0 ucf101 new 8
```

---

## Project Structure

```
FedDnC/
├── federated_main.py                  # Main entry point
├── requirements.txt
├── DATASETS.md
│
├── trainers/
│   ├── feddnc.py                      # ★ FedDnC trainer (dual-stream prompt learning)
│   └── fedmgp.py, fedpgp.py, ...      # Baseline trainers
│
├── federated_core/
│   ├── base_federated_learner.py      # Abstract federated training loop
│   ├── factory.py                     # Learner factory (name → class mapping)
│   ├── fed_utils.py                   # FedAvg, aggregation utilities
│   └── trainers/
│       ├── feddnc_learner.py          # ★ FedDnC aggregation (FedAvg + EMA)
│       └── ...                        # Baseline learners
│
├── configs/
│   ├── datasets/                      # 15 dataset YAML configs
│   └── trainers/
│       ├── FedDnC/                    # ★ FedDnC configs (5 experiment settings)
│       │   ├── base2novel_vit_b16.yaml
│       │   ├── domainnet_dir_alpha0.3.yaml
│       │   ├── domainnet_lodo_vit_b16.yaml
│       │   ├── office_dir_alpha0.3.yaml
│       │   └── office_lodo_vit_b16.yaml
│       └── FedPGP/, FedMGP/, ...      # Baseline configs
│
├── scripts/
│   ├── FedDnC/                        # ★ Train/eval scripts
│   │   ├── base2novel_train.sh / test.sh
│   │   ├── domainnet_train.sh / test.sh / lodo.sh
│   │   └── office_train.sh / lodo.sh
│   └── FedPGP/, FedMGP/, ...          # Baseline scripts
│
├── datasets/                          # Dataset loader classes (15 datasets)
├── clip/                              # CLIP with prompt support
└── Dassl/                             # Deep learning toolkit (Dassl.pytorch)
```

---

## Implementation Notes

### Trainer: `trainers/feddnc.py`

```python
class FedDnCPromptLearner(nn.Module):
    # P_b: randomly initialized discriminative stream
    self.ctx_b = nn.Parameter(ctx_vectors.clone())

    # P_n: warm-started from "a photo of a" token embeddings
    self.ctx_n = nn.Parameter(ctx_n_init)   # already near zero-shot optimum
```

### Federated Learner: `federated_core/trainers/feddnc_learner.py`

```python
# P_b: standard FedAvg
global_ctx_b = average_weights(client_ctx_b, client_ids, data_sizes)

# P_n: EMA (α=0.9 by default, adaptive if EMA_ALPHA=-1)
new_ctx_n = (1 - ema_n) * prev_ctx_n + ema_n * fedavg_ctx_n
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{fednc_icdm2026,
  title     = {Divide and Conquer: Federated Prompt Learning via Dual-Stream
               Prompts for Vision-Language Models},
  author    = {Anonymous},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

---

## Acknowledgements

- [CoOp / CoCoOp](https://github.com/KaiyangZhou/CoOp) — prompt learning framework
- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) — deep learning toolkit
- [CLIP](https://github.com/openai/CLIP) — vision-language backbone
- [FedPGP](https://github.com/HaokunChen245/FedPGP) — federated prompt baseline

---

## License

MIT License
