# GenomeOcean Classifier Distillation and Compression

Compression experiments for the GenomeOcean-based DNA sequence classifier. A fine-tuned 100M Mistral sequence classifier serves as the teacher for smaller students that predict **Cellular**, **NCLDV**, or **Phage**. This directory contains saved fold 1 models and training artifacts for the v3, 5 kb experiments.

## Repository scope

Training scripts live in this repository under `scripts/`. The shared evaluation script remains in the parent `finetuning_go_main` project:

- `scripts/train_kd.py`: randomly initialized students with logit distillation.
- `scripts/train_layerdrop.py`: students initialized from selected teacher layers, with logit distillation.
- `scripts/train_feature_cosine.py`: students initialized from selected teacher layers, with intermediate feature distillation.
- `scripts/train_ce.py`: a randomly initialized, hard-label-only baseline.
- `../scripts/test.py`: shared evaluation pipeline.

This repository includes the compression training scripts and saved artifacts. Training requires prepared datasets and a teacher checkpoint; evaluation also requires the parent project's `scripts/test.py`.

## Saved experiments

Architecture values below come from each saved `fold1/config.json`. Experiment names such as `8M`, `10M`, and `50M` are run tags rather than verified parameter counts.

| Directory | Objective | Initialization | Layers | Hidden size | FFN size | Attention / KV heads |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `kd_v3_50M_5kb` | CE + logit KL | Random | 6 | 768 | 2304 | 8 / 8 |
| `kd_v3_8M_5kb` | CE + logit KL | Random | 6 | 288 | 864 | 6 / 6 |
| `v3_layerdrop6L_5kb` | CE + logit KL | Teacher layer copying | 6 | 768 | 3072 | 8 / 8 |
| `v3_featcos6L_5kb` | CE + feature cosine distance | Teacher layer copying | 6 | 768 | 3072 | 8 / 8 |
| `ce_v3_10M_5kb` | CE only | Random | 4 | 256 | 512 | 4 / 4 |

All saved configurations use a vocabulary size of 4096 and the label mapping `0: Cellular`, `1: NCLDV`, `2: Phage`.

## Distillation methods

**Logit distillation** (`train_kd.py` and `train_layerdrop.py`) combines supervised cross-entropy with the teacher's softened class probabilities:

```text
p_teacher = softmax(teacher_logits / T)
p_student = softmax(student_logits / T)
loss = alpha * CE(student_logits, labels)
     + (1 - alpha) * T^2 * KL(p_teacher || p_student)
```

The implementation uses `torch.nn.functional.kl_div` with student log-probabilities, teacher probabilities, and `reduction="batchmean"`. The teacher is frozen and evaluated without gradients. Defaults are `alpha=0.5` and `T=4.0`.

**Layer dropping** reduces depth while keeping the teacher's width. The saved runs select zero-based teacher layers `[0, 2, 4, 7, 9, 11]` from a 12-layer teacher and map them to student layers `0–5`. Embeddings, final normalization, the classifier head, and selected transformer blocks are copied from the teacher. The mapping is recorded in `layer_selection.json`.

**Feature cosine distillation** uses the same layer-copying initialization but replaces logit KL with intermediate feature matching. It averages `1 - cosine_similarity` across non-padding tokens and matched layer pairs:

```text
loss = alpha * CE(student_logits, labels)
     + (1 - alpha) * mean_masked_feature_cosine_distance
```

Its default `alpha` is `0.5`; this objective has no temperature term. The CE baseline uses hard labels only and does not run a teacher forward pass.

## Requirements and data

Use a Python environment with `torch`, `transformers`, `datasets`, `huggingface_hub`, and `matplotlib`. Evaluation additionally uses `numpy`, `pandas`, and `scikit-learn`. Dependency versions are not pinned here. Training enables BF16 through `TrainingArguments`, so use a compatible GPU and PyTorch installation.

The parent project must provide:

- A teacher checkpoint with model weights, configuration, and tokenizer. The recorded runs used `ft_models/train_v3_100M_5kb/fold1/checkpoint-326785`.
- `data_5KB/fold1/train_v3.csv` with `sequence` and integer `label` columns.
- Prepared test data for the shared evaluation script.

The teacher tokenizer is reused for all students. The `5kb` setting truncates to 1250 tokens; `10kb` uses `data_10KB` and 2500 tokens. These are token limits, distinct from the DNA fragment length in base pairs.

## Train

Run the following from the parent `finetuning_go_main` directory. The training scripts default to the parent project as their working directory; the commands below also set it explicitly.

```bash
cd /mnt/taskmaster1/scratch/hyejong/01_gv_genomeocean_5fold/finetuning_go_main
WORK_DIR="$PWD"
TEACHER_PATH="$WORK_DIR/ft_models/train_v3_100M_5kb/fold1/checkpoint-326785"

# Randomly initialized 50M-tag student
CUDA_VISIBLE_DEVICES=0 python compression/scripts/train_kd.py \
  --work_dir "$WORK_DIR" --teacher_model_path "$TEACHER_PATH" \
  --student_tag 50M --data_version v3 --size_tag 5kb

# Randomly initialized 8M-tag student
CUDA_VISIBLE_DEVICES=0 python compression/scripts/train_kd.py \
  --work_dir "$WORK_DIR" --teacher_model_path "$TEACHER_PATH" \
  --student_tag 8M --hidden_size 288 --intermediate_size 864 \
  --num_hidden_layers 6 --num_attention_heads 6 --num_key_value_heads 6 \
  --data_version v3 --size_tag 5kb

# Layer dropping with logit distillation
CUDA_VISIBLE_DEVICES=0 python compression/scripts/train_layerdrop.py \
  --work_dir "$WORK_DIR" --teacher_model_path "$TEACHER_PATH" \
  --student_tag layerdrop6L --student_num_hidden_layers 6 \
  --data_version v3 --size_tag 5kb --fold 1

# Layer dropping with feature cosine distillation
CUDA_VISIBLE_DEVICES=0 python compression/scripts/train_feature_cosine.py \
  --work_dir "$WORK_DIR" --teacher_model_path "$TEACHER_PATH" \
  --student_tag featcos6L --student_num_hidden_layers 6 \
  --data_version v3 --size_tag 5kb --fold 1

# Hard-label-only baseline
CUDA_VISIBLE_DEVICES=0 python compression/scripts/train_ce.py \
  --work_dir "$WORK_DIR" --base_model_path "$TEACHER_PATH" \
  --student_tag 10M --data_version v3 --size_tag 5kb --folds 1
```

KD defaults are 5 epochs, batch size 8, gradient accumulation 8, learning rate `3e-5`, and checkpoint saving every 2000 optimizer steps. The CE baseline defaults to learning rate `3e-4`. On a single device, the effective batch size is 64. Training automatically resumes from the latest checkpoint in the output directory, so use a new student tag for a separate run.

`train_kd.py` trains fold 1 only. Layer-dropping and feature-cosine scripts accept `--fold`; the CE script accepts `--folds` and defaults to all five folds. The artifacts currently present here cover fold 1 only. Training does not perform validation or select a best model by validation score.

New outputs are written under `ft_models/`, not directly into `compression/`:

- Logit/feature KD: `ft_models/kd_v3_<student_tag>_5kb/fold1/`.
- CE baseline: `ft_models/ce_v3_<student_tag>_5kb/fold1/`.

In particular, new layer-dropping and feature-cosine outputs have the `kd_` prefix, while their saved directories here are named `v3_layerdrop6L_5kb` and `v3_featcos6L_5kb`.

## Evaluate a saved student

With `WORK_DIR` set as above:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test.py \
  --work_dir "$WORK_DIR" \
  --model_dir "$WORK_DIR/compression/kd_v3_8M_5kb/fold1" \
  --model_tag kd_8M --data_version v3 --size_tag 5kb \
  --fold 1 --shard_size 60000
```

Change `--model_dir` and `--model_tag` to evaluate another student or a newly trained model under `ft_models/`. Evaluation needs only the student and its tokenizer. The shared pipeline predicts chunks, aggregates them by contig using majority vote, and writes predictions, a confusion matrix, and a classification report beneath `results/test_v3_<model_tag>_5kb/`. Test logs are written under `log/test/`.

## Artifacts

Each saved `fold1/` directory contains:

- `model.safetensors`, `config.json`, `configuration_mistral.py`, and tokenizer files for model loading.
- `student_final.pth`: an additional PyTorch state dictionary.
- `training_args.bin`: serialized training arguments.
- `train_loss.csv` and `train_loss.png`: training loss history and plot.
- `train_runtime.json`: runtime segments, dataset size, and peak CUDA memory measurements.
- `checkpoint-326790/`: a saved Trainer checkpoint with optimizer, scheduler, RNG, and Trainer state.
- `layer_selection.json` for the two layer-copying experiments.

Recorded metadata reports 4,182,873 training rows for each saved run. Training loss and runtime measurements do not establish classification accuracy or inference speed; use the evaluation outputs for predictive comparisons and separate benchmarks for inference performance.
