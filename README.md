# GradFix 🚀

[![arXiv](https://img.shields.io/badge/arXiv-2510.09658-b31b1b.svg?style=flat)](https://arxiv.org/abs/2510.09658) [![GitHub stars](https://img.shields.io/github/stars/fillo-rinaldi/GradFix?style=social)](https://github.com/fillo-rinaldi/GradFix/stargazers) [![BibTeX](https://img.shields.io/badge/BibTeX-Available-brightgreen.svg)](#citation)

This repository contains the official implementation for the ICLR 2026 paper:

> **[Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](https://arxiv.org/abs/2510.09658)**
>
> [Filippo Rinaldi](https://scholar.google.com/citations?hl=it&user=muSVBYMAAAAJ), [Aniello Panariello](https://scholar.google.com/citations?hl=it&user=YnVl5pQAAAAJ), [Giacomo Salici](https://scholar.google.com/citations?hl=it&user=jMVa-rYAAAAJ), [Fengyuan Liu](https://scholar.google.com/citations?hl=it&user=fUVOyM4AAAAJ), [Marco Ciccone](https://scholar.google.com/citations?hl=it&user=hOQjblcAAAAJ), [Angelo Porrello](https://scholar.google.com/citations?hl=it&user=pGNOOk0AAAAJ), [Simone Calderara](https://scholar.google.com/citations?hl=it&user=YaRuDkcAAAAJ)

GradFix provides a principled and practical framework for real-gradient-guided task vector transport and merging in OpenCLIP models.

It provides a practical framework to:
- build GradFix-masked task vectors from real image gradients,
- transfer task vectors across different pretraining backbones,
- evaluate merge strategies across datasets and pretraining sources.


## Key Features ✨

- 🔁 **GradFix with Real Gradients:** Compute gradient-sign masks from real images and apply them to task vectors.
- 🧠 **Task Vector Transfer:** Build and evaluate $\tau_A$ and GradFix variants between two backbones (A → B).
- 🧩 **Merging Strategies:** Support `mean` and `ties` merging, with both “merge then mask” and “mask then merge” pipelines.


## Installation ⚙️


Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (task-vector submodule dependencies):

```bash
conda env create -f task_vectors/environment.yml
```

### Fine-tuned Checkpoints

Experiments expect finetuned checkpoints under:

```text
<base_folder>/clip-finetuned-weights/<dataset>/<arch>/<pretraining_backbone>/best.pt
```

Generate checkpoints with the fine-tuning command shown below, adjusting dataset/backbone and output paths for your setup.


## Usage 🧪

### 1) Real-gradient evaluation (`--mode eval`) ⚡

Runs transfer from backbone A to backbone B and evaluates:
- source task vector (`tau_A_source`)
- oracle GradFix (`tau_A_GradFix_oracle`)
- real-gradient GradFix (`tau_A_GradFix_realgrad`)

```bash
python main.py --mode eval \
  --base_folder /path/to/base \
  --dataset resisc45 \
  --arch ViT-B-16 \
  --pretraining_backbone_A datacomp_xl_s13b_b90k \
  --pretraining_backbone_B laion2b_s34b_b88k \
  --real_imgs_per_class 10 \
  --sign_mode max \
  --mask_mode normal \
  --eval_alphas 10 \
  --wandb_mode offline
```

For real-gradient computation, these options are **mutually exclusive** (use at most one):
- `--precomputed_indices <path_or_spec>`
- `--real_imgs_per_class <K>`
- `--num_batches <N>`

Detailed behavior:

- `--precomputed_indices <path_or_spec>`
  - Most reproducible option for controlled comparisons.
  - Accepts either:
    - a direct `.pkl` file path containing `dict[class_id -> list[index]]`, or
    - a method spec: `coreset:K`, `herding:K`, or `k-medoid:K` (also `,` / `_` as separators).
  - If a method spec is provided, the code first looks for a matching precomputed file in `precomputed_indices/`; if missing, it computes indices and can cache them.

- `--real_imgs_per_class <K>`
  - Randomly samples `K` training images per class.
  - Good for quick experiments and ablations on data budget.
  - Controlled by `--seed` for reproducibility.

- `--num_batches <N>`
  - Uses the standard training dataloader and computes signs on the first `N` batches.
  - Useful when class-balanced index files are not available.

If none of the three options is set, the code falls back to the full training dataloader.


### 2) Merge across datasets (`--mode merge`) 🔀

Loads multiple dataset-specific task vectors (same pretraining A), merges them, and applies GradFix strategies.

```bash
python main.py --mode merge \
  --base_folder /path/to/base \
  --dataset resisc45 \
  --arch ViT-B-16 \
  --pretraining_backbone_A datacomp_xl_s13b_b90k \
  --pretraining_backbone_B laion2b_s34b_b88k \
  --merge_taskvectors \
  --merge_method ties \
  --merge_ties_fraction 0.2 \
  --real_imgs_per_class 10 \
  --wandb_mode offline
```


### 3) Merge across pretraining sources (`--mode merge-mix`) 🧬

Loads task vectors from multiple pretraining backbones on the same dataset, then evaluates merged/GradFix outputs.

```bash
python main.py --mode merge-mix \
  --base_folder /path/to/base \
  --dataset resisc45 \
  --arch ViT-B-16 \
  --pretraining_backbone_B laion2b_s34b_b88k \
  --merge_method mean \
  --real_imgs_per_class 10 \
  --wandb_mode offline
```


### 4) Fine-tuning OpenCLIP 🔧

```bash
python grad_sign/finetune_openCLIP.py \
  --model_arch ViT-B-16 \
  --pretraining laion2b_s34b_b88k \
  --dataset mnist \
  --num_steps 2000 \
  --lr 1e-5 \
  --batch_size 32 \
  --base_folder /path/to/base
```


## Example Datasets 🗂️

Supported dataset wrappers are implemented in [grad_sign/dataset](grad_sign/dataset).
Current experiments in this repository use datasets including EuroSAT, GTSRB, SVHN, RESISC45, DTD, SUN397, MNIST, and Cars.


## Project Structure 📁

- `main.py` — unified experiment entrypoint and mode dispatcher.
- `grad_sign/` — core package (evaluation, merging, GradFix utilities, datasets, models, finetuning).
- `task_vectors/` — task vector utilities and external reference code.
- `permutations/` — permutation artifacts/utilities.


## Reproducibility Notes 🔬

- Set `--seed` for deterministic sampling behavior.
- Use `--wandb_mode online|offline|disabled` and optional `--wandb_group` to control experiment tracking.
- Expected finetuned checkpoint layout:

```text
<base_folder>/clip-finetuned-weights/<dataset>/<arch>/<pretraining_backbone>/best.pt
```


## Citation 📚

If you use this code or ideas from the paper, please cite:

```bibtex
@inproceedings{rinaldi2026gradient,
  title={Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models},
  author={Filippo Rinaldi and Aniello Panariello and Giacomo Salici and Fengyuan Liu and Marco Ciccone and Angelo Porrello and Simone Calderara},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```


## Acknowledgments 🙏

This project builds on ideas and code from:
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Task Vectors](https://github.com/mlfoundations/task_vectors)
- [TransFusion](https://github.com/aimagelab/TransFusion)

