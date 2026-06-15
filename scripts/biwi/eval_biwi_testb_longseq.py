#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_streamingtalker")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Wav2Vec2Processor

from algorithms.models import get_model


DEFAULT_CHECKPOINT = "checkpoints/diffar_biwi_241230.ckpt"
DEFAULT_CFG = "configs/biwi/stage2_diffar.yaml"
DEFAULT_REFERENCE_ROOT = "/m2v_intern/yangyifan20/StreamingTalker/longseq_biwi_testb"
DEFAULT_PROTOCOL = "testB_long2000"

TRAIN_ID_TO_INDEX = {"F2": 0, "F3": 1, "F4": 2, "M3": 3, "M4": 4, "M5": 5}
BIWI_FPS = 25
MODEL_SR = 16000
BIWI_N_VERTS = 23370
BIWI_LIP_UPPER_IDS = np.array((366, 12773, 21688, 10235, 10117), dtype=np.int64)
BIWI_LIP_DOWNER_IDS = np.array((21538, 10665, 22962, 21152, 14028), dtype=np.int64)
HUBERT_KERNELS = [10, 3, 3, 3, 3, 2, 2]
HUBERT_STRIDES = [5, 2, 2, 2, 2, 2, 2]
METRIC_FIELDS = ["LVE", "FDD", "abs_FDD", "MOD", "eval_len", "pred_len"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_cfg(cfg_path: str, checkpoint: str, inference_steps: int | None = None) -> Any:
    cfg = OmegaConf.merge(OmegaConf.load("./configs/base.yaml"), OmegaConf.load(cfg_path))
    cfg.NAME = "biwi_testb_longseq_eval"
    cfg.ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.DEVICE = [0]
    cfg.DEBUG = False
    cfg.DEMO.CHECKPOINTS = checkpoint
    cfg.DEMO.TEMPLATE = "./data/biwi/templates.pkl"
    cfg.DEMO.PLY = "./data/biwi/templates/BIWI.ply"
    cfg.DEMO.FPS = BIWI_FPS
    cfg.TEST.CHECKPOINTS = checkpoint
    if inference_steps is not None:
        cfg.MODEL.num_inference_timesteps = int(inference_steps)
    return cfg


def load_templates(path: str = "./data/biwi/templates.pkl") -> dict[str, np.ndarray]:
    with open(path, "rb") as fin:
        return pickle.load(fin, encoding="latin1")


def hubert_feature_len(num_samples: int) -> int:
    length = int(num_samples)
    for kernel, stride in zip(HUBERT_KERNELS, HUBERT_STRIDES):
        length = (length - kernel) // stride + 1
    return max(length, 0)


def predicted_len_from_samples(num_samples: int) -> int:
    return int(hubert_feature_len(num_samples) * (BIWI_FPS / 50))


def split_ranges(total_samples: int, chunk_samples: int) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_samples)
        ranges.append((start, end))
        start = end
    if len(ranges) > 1 and predicted_len_from_samples(ranges[-1][1] - ranges[-1][0]) < 2:
        prev_start, _ = ranges[-2]
        _, last_end = ranges[-1]
        ranges = ranges[:-2] + [(prev_start, last_end)]
    return ranges


def load_region(path: Path) -> list[int]:
    return [int(i) for i in path.read_text().strip().split(", ") if i]


def compute_biwi_metrics(gt_flat: np.ndarray, pred_flat: np.ndarray, template_flat: np.ndarray) -> dict[str, float]:
    mouth_map = load_region(Path("data/biwi/regions/lve.txt"))
    upper_map = load_region(Path("data/biwi/regions/fdd.txt"))
    gt = gt_flat.reshape(-1, BIWI_N_VERTS, 3)
    pred = pred_flat.reshape(-1, BIWI_N_VERTS, 3)
    template = template_flat.reshape(1, BIWI_N_VERTS, 3)
    eval_len = min(gt.shape[0], pred.shape[0])
    gt = gt[:eval_len]
    pred = pred[:eval_len]

    mouth_sq = np.array([np.square(gt[:, v, :] - pred[:, v, :]) for v in mouth_map])
    mouth_sq = np.transpose(mouth_sq, (1, 0, 2))
    lve = float(np.max(np.sum(mouth_sq, axis=2), axis=1).mean())

    def upper_variance(motion: np.ndarray) -> float:
        upper_sq = np.array([np.square(motion[:, v, :]) for v in upper_map])
        upper_sq = np.transpose(upper_sq, (1, 0, 2))
        return float(np.mean(np.std(np.sum(upper_sq, axis=2), axis=0)))

    fdd = upper_variance(gt - template) - upper_variance(pred - template)
    pred_lip = np.linalg.norm(pred[:, BIWI_LIP_UPPER_IDS, :] - pred[:, BIWI_LIP_DOWNER_IDS, :], axis=-1)
    gt_lip = np.linalg.norm(gt[:, BIWI_LIP_UPPER_IDS, :] - gt[:, BIWI_LIP_DOWNER_IDS, :], axis=-1)
    mod = float(np.mean(np.abs(pred_lip - gt_lip), axis=1).mean())
    return {"LVE": lve, "FDD": float(fdd), "abs_FDD": float(abs(fdd)), "MOD": mod, "eval_len": int(eval_len)}


def boundary_metrics(pred_flat: np.ndarray, chunk_lengths: list[int]) -> dict[str, float | None]:
    if not chunk_lengths or len(chunk_lengths) < 2:
        return {"boundary_l2_mean": None, "boundary_lip_l2_mean": None}
    pred = pred_flat.reshape(-1, BIWI_N_VERTS, 3)
    boundaries = np.cumsum(chunk_lengths)[:-1]
    l2_values = []
    lip_values = []
    lip_ids = BIWI_LIP_UPPER_IDS.tolist() + BIWI_LIP_DOWNER_IDS.tolist()
    for boundary in boundaries:
        if boundary <= 0 or boundary >= pred.shape[0]:
            continue
        jump = pred[boundary] - pred[boundary - 1]
        l2_values.append(float(np.linalg.norm(jump)))
        lip_values.append(float(np.linalg.norm(jump[lip_ids])))
    return {
        "boundary_l2_mean": float(np.mean(l2_values)) if l2_values else None,
        "boundary_lip_l2_mean": float(np.mean(lip_values)) if lip_values else None,
    }


def manifest_path(root: Path, protocol: str) -> Path:
    return root / "concat_dataset" / f"manifest_{protocol}.json"


def read_manifest(root: Path, protocol: str) -> dict[str, Any]:
    return read_json(manifest_path(root, protocol))


def source_sentence_ids(clip: dict[str, Any]) -> list[int]:
    return [int(seg["clip"].split("_e")[-1]) for seg in clip.get("source_segments", [])]


def selected_clip_subset(manifest: dict[str, Any], clip_names: list[str] | None) -> list[dict[str, Any]]:
    if not clip_names:
        return list(manifest["clips"])
    keep = set(clip_names)
    return [clip for clip in manifest["clips"] if clip["name"] in keep or clip["subject"] in keep]


def args_cfg_path(cfg: Any) -> str:
    return str(getattr(cfg, "_source_cfg_path", DEFAULT_CFG))


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        agg_row = {name: value for name, value in zip(keys, key)}
        agg_row["rows"] = len(group)
        agg_row["num_subjects"] = len({row["subject"] for row in group})
        agg_row["subjects"] = ",".join(sorted({row["subject"] for row in group}))
        for metric in METRIC_FIELDS:
            vals = [row.get(metric) for row in group if row.get(metric) is not None and row.get(metric) != ""]
            vals_f = [float(value) for value in vals]
            agg_row[f"{metric}_mean"] = float(np.mean(vals_f)) if vals_f else None
            agg_row[f"{metric}_std"] = float(np.std(vals_f, ddof=0)) if vals_f else None
        out.append(agg_row)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(str(checkpoint), map_location="cpu")
    return payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload


def checkpoint_id_dim(state: dict[str, torch.Tensor]) -> int:
    key = "diff_ar_network.obj_vector.weight"
    if key not in state:
        raise SystemExit(f"Checkpoint is missing {key!r}; cannot infer identity count.")
    return int(state[key].shape[0])


def setup_model(args: argparse.Namespace, state: dict[str, torch.Tensor], id_dim: int) -> tuple[Any, torch.nn.Module, Wav2Vec2Processor, dict[str, np.ndarray], torch.device]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg, args.checkpoint, args.inference_steps)
    setattr(cfg, "_source_cfg_path", args.cfg)
    cfg.MODEL.DIFF_AR_DEC.id_dim = int(id_dim)
    if args.history_len is not None:
        cfg.MODEL.INFERENCE_HISTORY_LEN = int(args.history_len)
    cfg.DEMO.CHECKPOINTS = str(args.checkpoint)
    cfg.TEST.CHECKPOINTS = str(args.checkpoint)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    templates = load_templates(cfg.DEMO.TEMPLATE)
    model = get_model(cfg=cfg)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"Warning: missing keys while loading checkpoint: {len(incompatible.missing_keys)}", flush=True)
    if incompatible.unexpected_keys:
        print(f"Warning: unexpected keys while loading checkpoint: {len(incompatible.unexpected_keys)}", flush=True)
    model.to(device)
    model.eval()
    return cfg, model, processor, templates, device


def predict_samples_id_index(
    model: torch.nn.Module,
    samples: np.ndarray,
    id_index: int,
    template: np.ndarray,
    processor: Wav2Vec2Processor,
    cfg: Any,
    device: torch.device,
) -> np.ndarray:
    input_values = np.squeeze(processor(samples, sampling_rate=MODEL_SR).input_values)
    audio = torch.as_tensor(input_values.reshape(1, -1), dtype=torch.float32, device=device)
    template_t = torch.as_tensor(template.reshape(-1), dtype=torch.float32, device=device)
    one_hot = torch.zeros((1, cfg.MODEL.DIFF_AR_DEC.id_dim), dtype=torch.float32, device=device)
    one_hot[0, int(id_index)] = 1.0
    with torch.no_grad():
        pred = model.predict({"id": one_hot, "audio": audio, "template": template_t, "vertice": None})
    return pred.squeeze(0).detach().cpu().numpy()


def adjust_prediction_len(pred: np.ndarray, target_len: int | None) -> tuple[np.ndarray, str, int]:
    raw_len = int(pred.shape[0])
    if target_len is None or raw_len == int(target_len):
        return pred, "none", raw_len
    if raw_len > int(target_len):
        return pred[: int(target_len)], f"trim_to_direct_len_{target_len}", raw_len
    if raw_len == 0:
        raise ValueError("Cannot pad an empty prediction")
    pad = np.repeat(pred[-1:], int(target_len) - raw_len, axis=0)
    return np.concatenate([pred, pad], axis=0), f"pad_to_direct_len_{target_len}", raw_len


def meta_complete(meta_path: Path) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = read_json(meta_path)
        pred = meta.get("prediction")
        if pred and Path(pred).exists() and Path(pred).stat().st_size > 0:
            return True
        return bool(meta.get("metrics")) and meta.get("pred_len") is not None
    except Exception:
        return False


def meta_path_for(
    root: Path,
    *,
    identity_mode: str,
    protocol: str,
    seed: int,
    clip_name: str,
    id_label: str,
    chunk_frames: int,
    final_delta: int,
) -> Path:
    return (
        root
        / "predictions"
        / identity_mode
        / protocol
        / f"seed_{seed}"
        / f"chunk_{chunk_frames:04d}_delta_{final_delta:+d}"
        / id_label
        / f"{clip_name}_meta.json"
    )


def run_chunked_id(
    *,
    root: Path,
    identity_mode: str,
    protocol: str,
    clip: dict[str, Any],
    id_index: int,
    id_label: str,
    id_policy: str,
    id_selection_method: str,
    cfg: Any,
    model: torch.nn.Module,
    processor: Wav2Vec2Processor,
    templates: dict[str, np.ndarray],
    device: torch.device,
    chunk_frames: int,
    final_delta: int,
    seed: int,
    skip_existing: bool,
    save_chunks: bool,
    save_prediction_arrays: bool,
) -> dict[str, Any]:
    meta_path = meta_path_for(
        root,
        identity_mode=identity_mode,
        protocol=protocol,
        seed=seed,
        clip_name=clip["name"],
        id_label=id_label,
        chunk_frames=chunk_frames,
        final_delta=final_delta,
    )
    if skip_existing and meta_complete(meta_path):
        return read_json(meta_path)

    samples, _ = librosa.load(clip["wav"], sr=MODEL_SR, mono=True)
    direct_pred_len = int(clip.get("expected_direct_pred_len") or 0) or None
    chunk_samples = int(round(chunk_frames / BIWI_FPS * MODEL_SR))
    ranges = split_ranges(int(samples.shape[0]), chunk_samples)
    chunks_dir = meta_path.parent / "chunks"
    if save_chunks:
        chunks_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    chunks = []
    start_time = time.time()
    for idx, (start, end) in enumerate(ranges):
        chunk = samples[start:end]
        if idx == len(ranges) - 1 and final_delta != 0:
            if final_delta < 0:
                chunk = chunk[: max(1, len(chunk) + final_delta)]
            else:
                chunk = np.pad(chunk, (0, final_delta))
        pred = predict_samples_id_index(
            model,
            chunk.astype(np.float32, copy=False),
            id_index,
            templates[clip["subject"]],
            processor,
            cfg,
            device,
        )
        pred_path = chunks_dir / f"{clip['name']}_chunk_{idx:03d}.npy"
        if save_chunks:
            np.save(pred_path, pred.astype(np.float32, copy=False))
        preds.append(pred)
        chunks.append(
            {
                "chunk_idx": int(idx),
                "sample_start": int(start),
                "sample_end": int(end),
                "samples_raw": int(end - start),
                "samples_inferred": int(len(chunk)),
                "pred_len": int(pred.shape[0]),
                "prediction": str(pred_path) if save_chunks else None,
            }
        )

    concat = np.concatenate(preds, axis=0)
    pred, length_policy, raw_pred_len = adjust_prediction_len(concat, direct_pred_len)
    pred_path = meta_path.with_name(f"{clip['name']}_pred.npy")
    prediction_value = str(pred_path) if save_prediction_arrays else None
    if save_prediction_arrays:
        np.save(pred_path, pred.astype(np.float32, copy=False))

    gt = np.load(clip["vertices"])
    metrics = compute_biwi_metrics(gt, pred, templates[clip["subject"]].reshape(-1))
    chunk_lens = [int(c["pred_len"]) for c in chunks]
    if sum(chunk_lens) != pred.shape[0] and chunk_lens:
        chunk_lens[-1] += int(pred.shape[0]) - sum(chunk_lens)
    bmetrics = boundary_metrics(pred, chunk_lens)

    meta = {
        "protocol": protocol,
        "mode": "chunked",
        "identity_mode": identity_mode,
        "clip_name": clip["name"],
        "subject": clip["subject"],
        "source_sentence_ids": source_sentence_ids(clip),
        "id_policy": id_policy,
        "train_id_used": id_label,
        "id_selection_method": id_selection_method,
        "subject_id_index": int(id_index),
        "chunk_frames": int(chunk_frames),
        "chunk_seconds": float(chunk_frames / BIWI_FPS),
        "length_policy": length_policy if length_policy != "none" else "last_chunk_adjusted",
        "final_delta_samples": int(final_delta),
        "wav": clip["wav"],
        "gt_vertices": clip["vertices"],
        "prediction": prediction_value,
        "gt_len": int(clip["frames"]),
        "pred_len": int(pred.shape[0]),
        "raw_pred_len": int(raw_pred_len),
        "direct_pred_len": int(direct_pred_len) if direct_pred_len is not None else None,
        "num_chunks": int(len(chunks)),
        "num_inference_timesteps": int(cfg.MODEL.num_inference_timesteps),
        "inference_history_len": int(cfg.MODEL.get("INFERENCE_HISTORY_LEN", cfg.MODEL.window_size)),
        "seed": int(seed),
        "checkpoint": str(cfg.DEMO.CHECKPOINTS),
        "cfg": args_cfg_path(cfg),
        "runtime_sec": float(time.time() - start_time),
        "save_chunks": bool(save_chunks),
        "save_prediction_arrays": bool(save_prediction_arrays),
        "chunks": chunks,
        "metrics": metrics,
        "boundary_metrics": bmetrics,
    }
    write_json(meta_path, meta)
    print(
        f"{identity_mode}: {clip['name']} id={id_label} row={id_index} "
        f"LVE={metrics['LVE']:.8g} FDD={metrics['FDD']:.8g} MOD={metrics['MOD']:.8g} "
        f"pred_len={pred.shape[0]} runtime={meta['runtime_sec']:.2f}s",
        flush=True,
    )
    return meta


def row_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    metrics = meta.get("metrics", {})
    bmetrics = meta.get("boundary_metrics", {})
    return {
        "protocol": meta.get("protocol"),
        "subject": meta.get("subject"),
        "clip_name": meta.get("clip_name"),
        "source_sentence_ids": ",".join(str(x) for x in meta.get("source_sentence_ids", [])),
        "mode": meta.get("mode"),
        "identity_mode": meta.get("identity_mode"),
        "id_policy": meta.get("id_policy"),
        "train_id_used": meta.get("train_id_used"),
        "id_selection_method": meta.get("id_selection_method"),
        "subject_id_index": meta.get("subject_id_index"),
        "chunk_frames": meta.get("chunk_frames"),
        "length_policy": meta.get("length_policy"),
        "final_delta_samples": meta.get("final_delta_samples"),
        "eval_len": metrics.get("eval_len"),
        "pred_len": meta.get("pred_len"),
        "raw_pred_len": meta.get("raw_pred_len"),
        "direct_pred_len": meta.get("direct_pred_len"),
        "LVE": metrics.get("LVE"),
        "FDD": metrics.get("FDD"),
        "abs_FDD": metrics.get("abs_FDD"),
        "MOD": metrics.get("MOD"),
        "boundary_l2_mean": bmetrics.get("boundary_l2_mean"),
        "boundary_lip_l2_mean": bmetrics.get("boundary_lip_l2_mean"),
        "seed": meta.get("seed"),
        "checkpoint": meta.get("checkpoint"),
        "cfg": meta.get("cfg"),
        "runtime_sec": meta.get("runtime_sec"),
        "num_chunks": meta.get("num_chunks"),
        "prediction": meta.get("prediction"),
    }


def ids_for_clip() -> list[tuple[int, str, str, str]]:
    return [
        (idx, train_id, "repo_unknown_mean", "repo_test_step_all_train_ids_mean_sequential")
        for train_id, idx in TRAIN_ID_TO_INDEX.items()
    ]


def write_reports(root: Path, rows: list[dict[str, Any]], identity_mode: str) -> None:
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "metrics_testb_longseq.csv", rows)
    agg = aggregate(
        rows,
        ["protocol", "mode", "identity_mode", "id_policy", "chunk_frames", "length_policy", "final_delta_samples", "seed"],
    )
    per_subject = aggregate(
        rows,
        ["protocol", "mode", "identity_mode", "id_policy", "subject", "chunk_frames", "length_policy", "final_delta_samples", "seed"],
    )
    per_id = aggregate(
        rows,
        ["protocol", "mode", "identity_mode", "id_policy", "train_id_used", "chunk_frames", "length_policy", "final_delta_samples", "seed"],
    )
    write_csv(report_dir / "metrics_testb_longseq_aggregate.csv", agg)
    write_csv(report_dir / "metrics_testb_longseq_by_subject.csv", per_subject)
    write_csv(report_dir / "metrics_testb_longseq_by_id.csv", per_id)
    write_json(
        report_dir / "metrics_testb_longseq.json",
        {"identity_mode": identity_mode, "rows": rows, "aggregate": agg},
    )

    lines = [
        "# BIWI Test-B Longseq Evaluation",
        "",
        f"- Identity mode: `{identity_mode}`",
        f"- Rows: `{len(rows)}`",
        "",
        "| identity mode | policy | chunk | delta | rows | subjects | LVE | FDD | MOD |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in agg:
        lve = float(row["LVE_mean"])
        fdd = float(row["FDD_mean"])
        mod = float(row["MOD_mean"])
        lines.append(
            f"| {row['identity_mode']} | {row['id_policy']} | {row['chunk_frames']} | "
            f"{row['final_delta_samples']} | {row['rows']} | {row['subjects']} | "
            f"{lve} | {fdd} | {mod} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- Metrics CSV: `{report_dir / 'metrics_testb_longseq.csv'}`",
        f"- Aggregate CSV: `{report_dir / 'metrics_testb_longseq_aggregate.csv'}`",
        f"- Per-subject CSV: `{report_dir / 'metrics_testb_longseq_by_subject.csv'}`",
        f"- Per-id CSV: `{report_dir / 'metrics_testb_longseq_by_id.csv'}`",
    ]
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote Test-B longseq reports to {report_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--chunk-frames", type=int, default=1000)
    parser.add_argument("--final-delta-samples", type=int, default=200)
    parser.add_argument("--history-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20240515)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--clips", nargs="+", default=None, help="Clip names or subjects, e.g. F1 M6")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-chunks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-prediction-arrays", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    state = load_checkpoint_state(args.checkpoint)
    id_dim = checkpoint_id_dim(state)
    identity_mode = "original6_unknown_mean"
    if id_dim < 6:
        raise SystemExit(f"{identity_mode} requires at least 6 identity rows, but checkpoint has {id_dim}")

    checkpoint_stem = Path(args.checkpoint).stem
    artifact_root = Path(args.artifact_root) if args.artifact_root else Path("longseq_artifacts_biwi") / f"{checkpoint_stem}_testb_longseq_{identity_mode}"

    cfg, model, processor, templates, device = setup_model(args, state, id_dim)
    manifest = read_manifest(Path(args.reference_root), args.protocol)
    rows: list[dict[str, Any]] = []
    for clip in selected_clip_subset(manifest, args.clips):
        for id_index, id_label, id_policy, id_selection_method in ids_for_clip():
            meta = run_chunked_id(
                root=artifact_root,
                identity_mode=identity_mode,
                protocol=args.protocol,
                clip=clip,
                id_index=id_index,
                id_label=id_label,
                id_policy=id_policy,
                id_selection_method=id_selection_method,
                cfg=cfg,
                model=model,
                processor=processor,
                templates=templates,
                device=device,
                chunk_frames=args.chunk_frames,
                final_delta=args.final_delta_samples,
                seed=args.seed,
                skip_existing=args.skip_existing,
                save_chunks=args.save_chunks,
                save_prediction_arrays=args.save_prediction_arrays,
            )
            rows.append(row_from_meta(meta))
    write_reports(artifact_root, rows, identity_mode)


if __name__ == "__main__":
    main()
