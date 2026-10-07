#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-dropping feature distillation training script.

Teacher:
    Fine-tuned GenomeOcean checkpoint for sequence classification

Student:
    Same width as teacher (hidden_size / attention heads / intermediate size)
    Fewer transformer layers only
    Initialized by uniformly sampling teacher layers and copying their weights

Loss:
    loss = alpha * CE(student, hard_labels)
         + (1 - alpha) * cosine_feature_loss(student_hidden, teacher_hidden)

Feature loss compares mapped intermediate hidden features with token-wise
cosine similarity, masked by attention_mask to ignore padding.

Example
-------
cd /mnt/taskmaster1/scratch/hyejong/01_gv_genomeocean_5fold/finetuning_go_main/compression/scripts
CUDA_VISIBLE_DEVICES=0 python train_feature_cosine.py
"""

import os
import sys
import json
import time
import glob
import math
import csv
import argparse
import datetime
import inspect
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            if past_key_values is None:
                return cls()

            cache = cls()
            for layer_idx, layer_past in enumerate(past_key_values):
                if layer_past is None:
                    continue
                if isinstance(layer_past, (list, tuple)) and len(layer_past) >= 2:
                    key_states, value_states = layer_past[0], layer_past[1]
                    cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache
        print("[INFO] Applied DynamicCache.from_legacy_cache compatibility patch")
except Exception as e:
    print(f"[WARN] DynamicCache compatibility patch skipped: {e}")

LABEL_NAMES = {0: "Cellular", 1: "NCLDV", 2: "Phage"}


class Tee:
    """Duplicate stdout/stderr to a .txt file."""

    def __init__(self, fpath):
        self.file = open(fpath, "a", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        self._last_pb_write = 0.0
        self._pb_interval_sec = 60.0

    def write(self, s):
        try:
            self._stdout.write(s)
            self._stdout.flush()
        except Exception:
            pass
        try:
            now = time.time()
            is_progress_update = ("\r" in s) and ("\n" not in s)
            if is_progress_update and (now - self._last_pb_write) < self._pb_interval_sec:
                return
            if is_progress_update:
                self._last_pb_write = now
            self.file.write(s)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._stdout.isatty()
        except Exception:
            return False

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        try:
            self.file.close()
        except Exception:
            pass


def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class RuntimeAccumulator:
    def __init__(self, out_json_path, meta=None):
        self.out_json_path = out_json_path
        self.meta = meta or {}
        self.t0 = None
        self.rows_this = 0
        self.bp_this = 0
        self.seg_started_at = None

    def start(self):
        self.t0 = time.time()
        self.seg_started_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def add_progress(self, rows_inc=0, bp_inc=0):
        self.rows_this += int(rows_inc)
        self.bp_this += int(bp_inc)

    def finish(self, stats: dict):
        t1 = time.time()
        seg_seconds = max(0.0, t1 - (self.t0 or t1))
        seg = {
            "started_at": self.seg_started_at,
            "finished_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds": seg_seconds,
            "rows": int(self.rows_this),
            "approx_bp": int(self.bp_this),
        }
        for k, v in (stats or {}).items():
            if v is not None:
                seg[k] = int(v) if isinstance(v, (int, float)) else v

        acc = _load_json(self.out_json_path)
        if not acc:
            acc = {"meta": self.meta, "segments": []}
        acc.setdefault("segments", []).append(seg)

        total_seconds = sum(s.get("seconds", 0.0) for s in acc["segments"])
        total_rows = sum(s.get("rows", 0) for s in acc["segments"])
        total_bp = sum(s.get("approx_bp", 0) for s in acc["segments"])

        acc["total_seconds"] = total_seconds
        acc["train_rows"] = total_rows
        acc["approx_train_bp"] = total_bp
        acc["sec_per_kb"] = (total_seconds / (total_bp / 1000.0)) if total_bp else None

        def _seg_max(key):
            vals = [s.get(key, 0) for s in acc["segments"] if s.get(key) is not None]
            return int(max(vals)) if vals else 0

        acc["vram_peak_reserved_bytes_max"] = _seg_max("vram_peak_reserved_bytes")
        acc["vram_peak_allocated_bytes_max"] = _seg_max("vram_peak_allocated_bytes")

        _save_json(self.out_json_path, acc)
        return acc


def _read_loss_csv(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, "r") as f:
            _ = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                rows.append(
                    {
                        "global_step": int(float(parts[0])),
                        "epoch": float(parts[1]),
                        "loss": float(parts[2]),
                        "grad_norm": (None if parts[3] == "" else float(parts[3])),
                        "learning_rate": (None if parts[4] == "" else float(parts[4])),
                        "progress_pct": (None if parts[5] == "" else float(parts[5])),
                        "ts": parts[6],
                    }
                )
    except Exception:
        return rows
    return rows


def _write_loss_csv(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("global_step,epoch,loss,grad_norm,learning_rate,progress_pct,ts\n")
        for r in rows:
            def fmt(x):
                return "" if x is None else str(x)

            f.write(
                f"{r['global_step']},{r['epoch']},{r['loss']},"
                f"{fmt(r.get('grad_norm'))},{fmt(r.get('learning_rate'))},"
                f"{fmt(r.get('progress_pct'))},{r['ts']}\n"
            )


def _plot_loss_png(png_path, rows):
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: (r["epoch"], r["global_step"]))
    xs = [r["epoch"] for r in rows_sorted]
    ys = [r["loss"] for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", markersize=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Feature KD Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


class DinoStyleLogCallback(TrainerCallback):
    def __init__(self, out_dir, plot_every_n_logs: int = 1):
        self.out_dir = out_dir
        self.csv_path = os.path.join(out_dir, "train_loss.csv")
        self.png_path = os.path.join(out_dir, "train_loss.png")
        self.rows = _read_loss_csv(self.csv_path)
        self.by_step = {int(r["global_step"]): r for r in self.rows}
        self.plot_every_n_logs = max(1, int(plot_every_n_logs))
        self._log_counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if ("loss" not in logs) or ("epoch" not in logs):
            return

        gs = int(getattr(state, "global_step", 0))
        max_steps = int(getattr(state, "max_steps", 0)) if getattr(state, "max_steps", None) else 0
        ep = float(logs.get("epoch", 0.0))
        loss = float(logs.get("loss", 0.0))
        grad_norm = logs.get("grad_norm", None)
        lr = logs.get("learning_rate", None)
        progress_pct = (gs / max_steps * 100.0) if max_steps > 0 else None

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.by_step[gs] = {
            "global_step": gs,
            "epoch": ep,
            "loss": loss,
            "grad_norm": (None if grad_norm is None else float(grad_norm)),
            "learning_rate": (None if lr is None else float(lr)),
            "progress_pct": progress_pct,
            "ts": ts,
        }
        self.rows = list(self.by_step.values())
        _write_loss_csv(self.csv_path, self.rows)

        step_str = f"{gs}/{max_steps}" if max_steps > 0 else f"{gs}/NA"
        prog_str = f"{progress_pct:.1f}%" if progress_pct is not None else "NA%"
        gn_str = f"{float(grad_norm):.4f}" if grad_norm is not None else "NA"
        lr_str = f"{float(lr):.3e}" if lr is not None else "NA"
        print(
            f"[LOG] {prog_str} | step {step_str} | epoch {ep:.2f} | "
            f"loss {loss:.4f} | grad_norm {gn_str} | lr {lr_str}"
        )

        self._log_counter += 1
        if (self._log_counter % self.plot_every_n_logs) == 0:
            try:
                _plot_loss_png(self.png_path, self.rows)
            except Exception as e:
                print(f"[WARN] Plotting failed (ignored): {e}")
                try:
                    plt.close("all")
                except Exception:
                    pass


def paths(work_dir: str, size_tag: str):
    if size_tag.lower() == "5kb":
        return os.path.join(work_dir, "data_5KB"), 1250
    return os.path.join(work_dir, "data_10KB"), 2500


def resolve_checkpoint_path(model_path: str):
    if not model_path:
        return model_path

    if any(
        os.path.exists(os.path.join(model_path, fname))
        for fname in ("config.json", "pytorch_model.bin", "model.safetensors")
    ):
        return model_path

    try:
        last = get_last_checkpoint(model_path)
    except Exception:
        last = None

    if last and os.path.isdir(last):
        return last

    cands = glob.glob(os.path.join(model_path, "checkpoint-*"))
    if cands:
        cands.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]), reverse=True)
        return cands[0]

    raise FileNotFoundError(f"No model checkpoint found under: {model_path}")


def load_train_split(fold_dir: str, ver: str, max_rows: int = 0):
    train_csv = os.path.join(fold_dir, f"train_{ver}.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    if max_rows and max_rows > 0:
        rows = []
        with open(train_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if "label" in row:
                    row["label"] = int(row["label"])
                rows.append(row)
                if (idx + 1) >= max_rows:
                    break
        if not rows:
            raise ValueError(f"No rows loaded from {train_csv}")
        return DatasetDict({"train": Dataset.from_list(rows)})

    ds = load_dataset("csv", data_files={"train": train_csv})
    assert {"sequence", "label"}.issubset(ds["train"].column_names), "CSV must have sequence and label columns"
    return ds


def tokenize_dataset(ds, tok, max_len: int):
    def _tok(batch):
        return tok(batch["sequence"], truncation=True, max_length=max_len, return_token_type_ids=False)

    x = ds["train"].map(
        _tok,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in ("sequence", "label")],
    )
    if "token_type_ids" in x.column_names:
        x = x.remove_columns("token_type_ids")
    x = x.rename_column("label", "labels")
    return x.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


def build_training_args(
    out_dir,
    num_train_epochs,
    per_device_train_batch_size,
    learning_rate,
    save_steps,
    grad_accum,
    logging_steps,
):
    sig = inspect.signature(TrainingArguments.__init__)
    base = dict(
        output_dir=out_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        save_steps=save_steps,
        save_total_limit=2,
        gradient_accumulation_steps=grad_accum,
        logging_steps=logging_steps,
    )
    if "logging_strategy" in sig.parameters:
        base["logging_strategy"] = "steps"
    if "disable_tqdm" in sig.parameters:
        base["disable_tqdm"] = True
    if "report_to" in sig.parameters:
        base["report_to"] = []
    if "remove_unused_columns" in sig.parameters:
        base["remove_unused_columns"] = False
    if "dataloader_pin_memory" in sig.parameters:
        base["dataloader_pin_memory"] = bool(torch.cuda.is_available())

    kwargs = {k: v for k, v in base.items() if k in sig.parameters}
    if "bf16" in sig.parameters:
        kwargs["bf16"] = bool(torch.cuda.is_available())
    return TrainingArguments(**kwargs)


def print_param_summary(model, name="model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] total={total:,}  trainable={trainable:,}  frozen={total - trainable:,}")


def _round_half_up(x: float):
    return int(math.floor(x + 0.5))


def select_uniform_layer_indices(teacher_num_layers: int, student_num_layers: int):
    if student_num_layers < 1:
        raise ValueError("student_num_layers must be >= 1")
    if student_num_layers > teacher_num_layers:
        raise ValueError(
            f"student_num_layers ({student_num_layers}) cannot exceed teacher_num_layers ({teacher_num_layers})"
        )
    if student_num_layers == 1:
        return [teacher_num_layers // 2]

    indices = [
        _round_half_up(i * (teacher_num_layers - 1) / (student_num_layers - 1))
        for i in range(student_num_layers)
    ]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Layer selection produced duplicates: {indices}")
    return indices


def get_backbone(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise AttributeError("Expected a Mistral-style sequence classification model with model.layers")


def initialize_student_from_teacher(teacher, student, teacher_layer_indices):
    teacher_backbone = get_backbone(teacher)
    student_backbone = get_backbone(student)

    if len(student_backbone.layers) != len(teacher_layer_indices):
        raise ValueError(
            f"student layers ({len(student_backbone.layers)}) != "
            f"teacher_layer_indices ({len(teacher_layer_indices)})"
        )

    student_backbone.embed_tokens.load_state_dict(teacher_backbone.embed_tokens.state_dict())
    student_backbone.norm.load_state_dict(teacher_backbone.norm.state_dict())

    if hasattr(teacher, "score") and hasattr(student, "score"):
        student.score.load_state_dict(teacher.score.state_dict())

    for student_idx, teacher_idx in enumerate(teacher_layer_indices):
        student_backbone.layers[student_idx].load_state_dict(teacher_backbone.layers[teacher_idx].state_dict())


def save_layer_selection(path, teacher_layer_indices, teacher_num_layers, student_num_layers):
    obj = {
        "selection_strategy": "uniform_linspace",
        "teacher_num_hidden_layers": teacher_num_layers,
        "student_num_hidden_layers": student_num_layers,
        "teacher_layer_indices": teacher_layer_indices,
        "student_to_teacher": [
            {"student_layer": s_idx, "teacher_layer": t_idx}
            for s_idx, t_idx in enumerate(teacher_layer_indices)
        ],
    }
    _save_json(path, obj)


def _hidden_from_layer_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class FeatureCosineKDTrainer(Trainer):
    """
    loss = alpha * CE(student_logits, hard_labels)
         + (1 - alpha) * mean_layer_token_cosine_distance(student_hidden, teacher_hidden)
    """

    def __init__(self, teacher, alpha: float, teacher_layer_indices, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.alpha = alpha
        self.teacher_layer_indices = list(teacher_layer_indices)
        self.student_layer_indices = list(range(len(self.teacher_layer_indices)))
        self._teacher_features = {}
        self._student_features = {}
        self._hook_handles = []
        self._first_batch_logged = False
        self._register_feature_hooks()

    def _register_feature_hooks(self):
        teacher_layers = get_backbone(self.teacher).layers
        student_layers = get_backbone(self.model).layers

        def make_hook(store, idx):
            def _hook(module, inputs, output):
                store[idx] = _hidden_from_layer_output(output)

            return _hook

        for student_idx, teacher_idx in zip(self.student_layer_indices, self.teacher_layer_indices):
            self._hook_handles.append(student_layers[student_idx].register_forward_hook(make_hook(self._student_features, student_idx)))
            self._hook_handles.append(teacher_layers[teacher_idx].register_forward_hook(make_hook(self._teacher_features, teacher_idx)))

    def _clear_feature_buffers(self):
        self._teacher_features.clear()
        self._student_features.clear()

    def _remove_feature_hooks(self):
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles = []

    def _masked_tokenwise_cosine_loss(self, student_hidden, teacher_hidden, attention_mask=None):
        student_hidden = student_hidden.float()
        teacher_hidden = teacher_hidden.float()
        cosine = F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)
        if attention_mask is None:
            return 1.0 - cosine.mean()

        mask = attention_mask.to(cosine.device).float()
        denom = mask.sum().clamp_min(1.0)
        return ((1.0 - cosine) * mask).sum() / denom

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        attention_mask = model_inputs.get("attention_mask")

        self._clear_feature_buffers()
        student_out = model(**model_inputs)
        student_logits = student_out.logits

        device = student_logits.device
        if next(self.teacher.parameters()).device != device:
            self.teacher = self.teacher.to(device)

        with torch.no_grad():
            _ = self.teacher(**model_inputs)

        missing_student = [idx for idx in self.student_layer_indices if idx not in self._student_features]
        missing_teacher = [idx for idx in self.teacher_layer_indices if idx not in self._teacher_features]
        if missing_student or missing_teacher:
            raise RuntimeError(
                f"Feature hooks missing outputs. student={missing_student} teacher={missing_teacher}"
            )

        ce_loss = F.cross_entropy(student_logits, labels)

        layer_feature_losses = []
        for student_idx, teacher_idx in zip(self.student_layer_indices, self.teacher_layer_indices):
            layer_feature_losses.append(
                self._masked_tokenwise_cosine_loss(
                    student_hidden=self._student_features[student_idx],
                    teacher_hidden=self._teacher_features[teacher_idx],
                    attention_mask=attention_mask,
                )
            )

        feature_loss = torch.stack(layer_feature_losses).mean()
        loss = self.alpha * ce_loss + (1.0 - self.alpha) * feature_loss

        if not self._first_batch_logged:
            layer_loss_str = ", ".join(
                f"S{s_idx}->T{t_idx}:{layer_loss.detach().item():.4f}"
                for (s_idx, t_idx), layer_loss in zip(
                    zip(self.student_layer_indices, self.teacher_layer_indices),
                    layer_feature_losses,
                )
            )
            print(
                f"[KD] First batch losses: CE={ce_loss.item():.4f}  "
                f"FeatureCos={feature_loss.item():.4f}  Total={loss.item():.4f}"
            )
            print(f"[KD] Layer feature losses: {layer_loss_str}")
            self._first_batch_logged = True

        self._clear_feature_buffers()
        return (loss, student_out) if return_outputs else loss

    def __del__(self):
        self._remove_feature_hooks()


def main():
    ap = argparse.ArgumentParser(
        description="Layer-dropping feature distillation: keep width, shrink depth, match intermediate features."
    )
    ap.add_argument(
        "--work_dir",
        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        help="Root of finetuning_go_main directory",
    )
    ap.add_argument(
        "--teacher_model_path",
        default=None,
        help="Path to teacher checkpoint or fold directory containing checkpoints",
    )
    ap.add_argument("--student_tag", default="featcos6L")
    ap.add_argument("--data_version", default="v3")
    ap.add_argument("--size_tag", default="5kb")
    ap.add_argument("--fold", type=int, default=1)

    ap.add_argument(
        "--student_num_hidden_layers",
        type=int,
        default=6,
        help="Student depth only. Width-related config is inherited from teacher.",
    )

    ap.add_argument("--alpha", type=float, default=0.5, help="Weight for CE loss. (1-alpha) is feature cosine loss")

    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument(
        "--log_every_fraction",
        type=float,
        default=0.1,
        help="Log every this fraction of an epoch (0.1 => 10 logs/epoch)",
    )
    ap.add_argument("--plot_every_n_logs", type=int, default=1)
    ap.add_argument(
        "--max_train_rows",
        type=int,
        default=0,
        help="Optional debug cap. 0 means use the full training CSV.",
    )
    args = ap.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    data_root, max_len = paths(args.work_dir, args.size_tag)
    fold_dir = os.path.join(data_root, f"fold{args.fold}")

    if args.teacher_model_path is None:
        args.teacher_model_path = os.path.join(
            args.work_dir,
            f"ft_models/train_{args.data_version}_100M_{args.size_tag}/fold{args.fold}/checkpoint-326785",
        )

    teacher_model_path = resolve_checkpoint_path(args.teacher_model_path)

    out_dir = os.path.join(
        args.work_dir,
        f"ft_models/kd_{args.data_version}_{args.student_tag}_{args.size_tag}/fold{args.fold}",
    )
    os.makedirs(out_dir, exist_ok=True)

    log_root = os.path.join(args.work_dir, "log/train")
    os.makedirs(log_root, exist_ok=True)

    log_path = os.path.join(
        log_root,
        f"kd_{args.data_version}_{args.student_tag}_{args.size_tag}_fold{args.fold}_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt",
    )

    tee = Tee(log_path)
    print(f"[INFO] Logging to {log_path}")
    print(f"[INFO] time_local: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] time_utc:   {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%SZ}")
    print(f"[INFO] teacher_model_path: {teacher_model_path}")
    print(f"[INFO] output_dir:         {out_dir}")
    print(f"[INFO] fold_dir:           {fold_dir}")
    print(f"[INFO] alpha={args.alpha}")
    print(f"[INFO] student_num_hidden_layers={args.student_num_hidden_layers}")
    if args.max_train_rows > 0:
        print(f"[INFO] max_train_rows={args.max_train_rows} (debug cap enabled)")

    try:
        print("[INFO] Loading tokenizer from teacher checkpoint...")
        tok = AutoTokenizer.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            local_files_only=True,
            model_max_length=max_len,
            use_fast=True,
            padding_side="right",
        )
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})

        print("[INFO] Loading teacher model...")
        teacher = AutoModelForSequenceClassification.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if hasattr(teacher.config, "use_cache"):
            teacher.config.use_cache = False
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print_param_summary(teacher, "teacher")

        teacher_backbone = get_backbone(teacher)
        teacher_num_layers = len(teacher_backbone.layers)
        teacher_layer_indices = select_uniform_layer_indices(
            teacher_num_layers=teacher_num_layers,
            student_num_layers=args.student_num_hidden_layers,
        )
        print(f"[INFO] teacher_num_hidden_layers={teacher_num_layers}")
        print(f"[INFO] selected_teacher_layers={teacher_layer_indices}")

        print("[INFO] Building student model with same width as teacher...")
        config = AutoConfig.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config.num_hidden_layers = args.student_num_hidden_layers
        config.num_labels = len(LABEL_NAMES)
        config.id2label = {i: LABEL_NAMES[i] for i in LABEL_NAMES}
        config.label2id = {v: k for k, v in LABEL_NAMES.items()}
        config.pad_token_id = tok.pad_token_id
        config.use_cache = False
        config.teacher_model_path = teacher_model_path
        config.teacher_num_hidden_layers = teacher_num_layers
        config.teacher_layer_indices = teacher_layer_indices
        config.distillation_mode = "intermediate_feature_cosine"

        student = AutoModelForSequenceClassification.from_config(
            config,
            trust_remote_code=True,
        )

        if len(tok) != student.get_input_embeddings().num_embeddings:
            student.resize_token_embeddings(len(tok))

        initialize_student_from_teacher(
            teacher=teacher,
            student=student,
            teacher_layer_indices=teacher_layer_indices,
        )
        print_param_summary(student, "student")

        layer_selection_json = os.path.join(out_dir, "layer_selection.json")
        save_layer_selection(
            path=layer_selection_json,
            teacher_layer_indices=teacher_layer_indices,
            teacher_num_layers=teacher_num_layers,
            student_num_layers=args.student_num_hidden_layers,
        )
        print(f"[INFO] Layer selection JSON: {layer_selection_json}")

        print(f"[INFO] Loading dataset from {fold_dir}...")
        ds = load_train_split(fold_dir, args.data_version, max_rows=args.max_train_rows)
        n_rows = len(ds["train"])
        print(f"[INFO] Train rows: {n_rows:,}")

        train_tok = tokenize_dataset(ds, tok, max_len)

        num_batches = math.ceil(n_rows / args.per_device_train_batch_size)
        steps_per_epoch = max(1, math.ceil(num_batches / args.gradient_accumulation_steps))
        frac = float(args.log_every_fraction)
        if frac <= 0:
            raise ValueError("--log_every_fraction must be > 0")
        logging_steps = max(1, int(round(steps_per_epoch * frac)))
        print(f"[INFO] steps_per_epoch≈{steps_per_epoch}, logging_steps={logging_steps}")

        tr_args = build_training_args(
            out_dir=out_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
            save_steps=args.save_steps,
            grad_accum=args.gradient_accumulation_steps,
            logging_steps=logging_steps,
        )

        cb = DinoStyleLogCallback(out_dir, plot_every_n_logs=args.plot_every_n_logs)
        trainer_kwargs = dict(
            model=student,
            args=tr_args,
            train_dataset=train_tok,
            data_collator=DataCollatorWithPadding(tokenizer=tok),
            callbacks=[cb],
            teacher=teacher,
            alpha=args.alpha,
            teacher_layer_indices=teacher_layer_indices,
        )
        trainer_sig = inspect.signature(Trainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tok
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = tok

        trainer = FeatureCosineKDTrainer(**trainer_kwargs)

        runtime_json = os.path.join(out_dir, "train_runtime.json")
        meta = {
            "task": "KD_Cellular_NCLDV_Phage_feature_cosine",
            "version": args.data_version,
            "teacher": teacher_model_path,
            "student_tag": args.student_tag,
            "size_tag": args.size_tag,
            "fold": args.fold,
            "alpha": args.alpha,
            "teacher_num_hidden_layers": teacher_num_layers,
            "student_num_hidden_layers": args.student_num_hidden_layers,
            "teacher_layer_indices": teacher_layer_indices,
            "feature_loss": "masked_tokenwise_cosine",
        }
        acc = RuntimeAccumulator(runtime_json, meta=meta)
        acc.start()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            last_ckpt = get_last_checkpoint(out_dir)
        except Exception:
            last_ckpt = None

        if last_ckpt and os.path.isdir(last_ckpt):
            print("=" * 70)
            print(f"[RESUME] Found checkpoint: {last_ckpt}")
            print("=" * 70)
            trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            print("[INFO] Starting fresh feature-cosine KD training")
            trainer.train()

        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
        config.save_pretrained(out_dir)

        pth_path = os.path.join(out_dir, "student_final.pth")
        torch.save(student.state_dict(), pth_path)
        print(f"[INFO] Saved student_final.pth -> {pth_path}")

        stats = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats["vram_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            stats["vram_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())

        kb_per_row = 5000 if args.size_tag.lower() == "5kb" else 10000
        acc.add_progress(rows_inc=n_rows, bp_inc=n_rows * kb_per_row)
        acc.finish(stats)

        print(f"[INFO] time_local_end: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"[INFO] time_utc_end:   {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%SZ}")
        print(f"[INFO] Done. Output -> {out_dir}")
        print(f"[INFO] Loss CSV: {os.path.join(out_dir, 'train_loss.csv')}")
        print(f"[INFO] Loss PNG: {os.path.join(out_dir, 'train_loss.png')}")
        print(f"[INFO] Runtime JSON: {runtime_json}")

    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        tee.close()


if __name__ == "__main__":
    main()
