#!/usr/bin/env python
"""
eval_languagebind_retrieval.py

LanguageBind evaluation for both Temporal-Only and Mixed Retrieval.
Uses global embeddings for similarity computation.

Usage:
    python eval_languagebind_retrieval.py \
        --videos_source /data1/kaisi/datasets/test_avsync \
        --dataset avsync \
        --num_distractors 50 \
        --max_videos 150
"""

import sys
import os
import random
import argparse
import warnings

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Prepend project paths so local modules win over same-named site-packages.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_BASELINES = os.environ.get("FATE_BASELINE_ROOT", "/data1/kaisi/sync/baselines")
BASELINES_LIB_ROOT = ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "baselines", "LanguageBind")) else os.path.dirname(EXTERNAL_BASELINES)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(BASELINES_LIB_ROOT, "baselines"))
sys.path.insert(0, os.path.join(BASELINES_LIB_ROOT, "baselines", "LanguageBind"))

from datasets import AvsyncDatasetCrossRetrieval, VGGSOundDatasetCrossRetrieval
from languagebind import LanguageBind, transform_dict
from languagebind.video.processing_video import load_and_transform_video


# ============================================================
# Metrics
# ============================================================

def recall_at_k(relevance, scores, k):
    if not isinstance(relevance, torch.Tensor):
        relevance = torch.as_tensor(relevance, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32, device=relevance.device)
    else:
        scores = scores.to(relevance.device)
    k = min(k, relevance.numel())
    if k == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    rel_k = relevance[order][:k]
    total_relevant = (relevance > 0).sum().item()
    if total_relevant == 0:
        return 0.0
    return (rel_k > 0).sum().item() / total_relevant


def ndcg_at_k(relevance, scores, k):
    if not isinstance(relevance, torch.Tensor):
        relevance = torch.as_tensor(relevance, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32, device=relevance.device)
    else:
        scores = scores.to(relevance.device)
    k = min(k, relevance.numel())
    if k == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    rel_k = relevance[order][:k]
    gains = torch.pow(2.0, rel_k) - 1.0
    discounts = torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
    dcg = (gains / discounts).sum()
    ideal_rel = torch.sort(relevance, descending=True).values[:k]
    ideal_gains = torch.pow(2.0, ideal_rel) - 1.0
    ideal_dcg = (ideal_gains / discounts).sum()
    if ideal_dcg <= 0:
        return 0.0
    return (dcg / ideal_dcg).item()


def calculate_sim_global(query_emb, value_emb):
    return torch.matmul(value_emb, query_emb)


# ============================================================
# Model Loading
# ============================================================

def load_model(device, cache_dir, video_ckpt, audio_ckpt):
    clip_type = {
        'video': video_ckpt,
        'audio': audio_ckpt,
    }
    print(f"Loading LanguageBind: video={video_ckpt}, audio={audio_ckpt}")
    model = LanguageBind(clip_type=clip_type, cache_dir=cache_dir)
    model = model.to(device)
    model.eval()

    modality_transform = {k: transform_dict[k](model.modality_config[k]) for k in clip_type.keys()}
    return model, modality_transform


# ============================================================
# Segment Loading / Processing
# ============================================================

def load_audio_segment_waveform(video_path, start_time, duration, out_sr=16000):
    """Load one audio segment from an mp4 file as waveform [1, T]."""
    try:
        container = av.open(video_path)
        if len(container.streams.audio) == 0:
            container.close()
            return None

        audio_stream = container.streams.audio[0]
        sr = audio_stream.rate
        timestamp = int(start_time / audio_stream.time_base)
        container.seek(timestamp, stream=audio_stream)

        all_samples = []
        for frame in container.decode(audio=0):
            if frame.time is None:
                continue
            if frame.time < start_time:
                continue
            if frame.time > start_time + duration:
                break
            arr = frame.to_ndarray()
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > 100:
                arr = arr.T
            all_samples.append(arr)
        container.close()

        if not all_samples:
            return None

        target_ch = max(
            {(a.shape[1] if a.ndim == 2 else 1) for a in all_samples},
            key=lambda c: sum(1 for a in all_samples if (a.shape[1] if a.ndim == 2 else 1) == c),
        )
        processed = []
        for a in all_samples:
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.shape[1] > target_ch:
                a = a[:, :target_ch]
            elif a.shape[1] < target_ch:
                a = np.pad(a, ((0, 0), (0, target_ch - a.shape[1])))
            processed.append(a)

        waveform = torch.from_numpy(np.concatenate(processed, axis=0)).float().T

        if sr != out_sr:
            waveform = torchaudio.functional.resample(waveform, sr, out_sr)
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]

        expected = max(1, int(duration * out_sr))
        if waveform.shape[1] > expected:
            waveform = waveform[:, :expected]
        elif waveform.shape[1] < expected:
            waveform = F.pad(waveform, (0, expected - waveform.shape[1]))

        if waveform.shape[1] == 0:
            return None
        return waveform
    except Exception:
        return None


def get_embeddings(sample, model, modality_transform, device, batch_size=8):
    """
    Extract LanguageBind global embeddings for all segments of one video.
    Returns v_emb [N, D], a_emb [N, D] (normalized).
    """
    video_path = sample["video_path"]
    segments = sample["segments"]

    v_cfg = model.modality_config['video'].vision_config
    a_cfg = model.modality_config['audio'].vision_config

    v_backend = v_cfg.video_decode_backend
    v_num_frames = v_cfg.num_frames
    a_sr = a_cfg.audio_sample_rate
    a_num_mel = a_cfg.num_mel_bins
    a_target_len = a_cfg.target_length

    video_processor = modality_transform['video']
    audio_processor = modality_transform['audio']

    all_v = []
    all_a = []

    with torch.no_grad():
        for start in range(0, len(segments), batch_size):
            batch_segs = segments[start:start + batch_size]
            v_inputs = []
            a_inputs = []

            for seg_start, seg_end in batch_segs:
                seg_start, seg_end = float(seg_start), float(seg_end)
                dur = max(0.001, seg_end - seg_start)

                try:
                    v = load_and_transform_video(
                        video_path,
                        video_processor.transform,
                        video_decode_backend=v_backend,
                        clip_start_sec=seg_start,
                        clip_end_sec=seg_end,
                        num_frames=v_num_frames,
                    )
                except Exception:
                    v = torch.zeros(3, v_num_frames, 224, 224, dtype=torch.float32)

                waveform = load_audio_segment_waveform(video_path, seg_start, dur, out_sr=a_sr)
                if waveform is None:
                    a = torch.zeros(3, a_num_mel, a_target_len, dtype=torch.float32)
                else:
                    try:
                        a = audio_processor.transform((waveform, a_sr))
                    except Exception:
                        a = torch.zeros(3, a_num_mel, a_target_len, dtype=torch.float32)

                v_inputs.append(v)
                a_inputs.append(a)

            v_batch = torch.stack(v_inputs, dim=0).to(device)
            a_batch = torch.stack(a_inputs, dim=0).to(device)

            inputs = {
                'video': {'pixel_values': v_batch},
                'audio': {'pixel_values': a_batch},
            }

            outputs = model(inputs)
            v_emb = F.normalize(outputs['video'], dim=-1)
            a_emb = F.normalize(outputs['audio'], dim=-1)

            all_v.append(v_emb.cpu())
            all_a.append(a_emb.cpu())

    if not all_v:
        return None, None

    v_emb = F.normalize(torch.cat(all_v, dim=0), dim=-1)
    a_emb = F.normalize(torch.cat(all_a, dim=0), dim=-1)
    return v_emb, a_emb


# ============================================================
# Phase 1: Pre-compute
# ============================================================

def precompute_all_embeddings(dataset, model, modality_transform, device, batch_size=8):
    all_v, all_a, all_samples = [], [], []
    failed = 0

    for idx in tqdm(range(len(dataset)), desc="Pre-computing LanguageBind embeddings"):
        sample = dataset[idx]
        try:
            v_emb, a_emb = get_embeddings(sample, model, modality_transform, device, batch_size)
            if v_emb is None or a_emb is None:
                failed += 1
                continue
            all_v.append(v_emb)
            all_a.append(a_emb)
            all_samples.append(sample)
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Warning: {sample['video_path']}: {e}")

    print(f"Embeddings computed: {len(all_samples)} ok, {failed} failed")
    return all_v, all_a, all_samples


# ============================================================
# Phase 2a: Temporal-Only
# ============================================================

def evaluate_temporal_only(all_v, all_a, all_samples, ks):
    metrics = _empty_metrics(ks)
    for i in range(len(all_samples)):
        rel = all_samples[i]["relevance"].float()
        gt = int(all_samples[i]["gt_index"])
        s_a2v = calculate_sim_global(all_a[i][gt], all_v[i])
        s_v2a = calculate_sim_global(all_v[i][gt], all_a[i])
        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(rel, s_a2v, k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(rel, s_a2v, k)
            metrics["recall_v2a"][k] += recall_at_k(rel, s_v2a, k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(rel, s_v2a, k)
        metrics["count"] += 1
    return metrics


# ============================================================
# Phase 2b: Mixed Retrieval
# ============================================================

def evaluate_mixed_retrieval(all_v, all_a, all_samples, num_distractors, ks,
                             seed=42, device=None):
    if device is None:
        device = torch.device("cpu")
    rng = random.Random(seed)
    n = len(all_samples)
    actual_M = min(num_distractors, n - 1)
    metrics = _empty_metrics(ks)

    for qi in tqdm(range(n), desc=f"Mixed retrieval (M={actual_M + 1} videos)"):
        sample = all_samples[qi]
        gt = int(sample["gt_index"])

        others = [i for i in range(n) if i != qi]
        distractors = rng.sample(others, actual_M)

        cv = [all_v[qi]]
        ca = [all_a[qi]]
        rl = list(sample["relevance"].numpy().astype(np.float32))

        for di in distractors:
            cv.append(all_v[di])
            ca.append(all_a[di])
            rl.extend([0.0] * all_v[di].shape[0])

        all_cv = torch.cat(cv, dim=0).to(device)
        all_ca = torch.cat(ca, dim=0).to(device)
        rel_t = torch.tensor(rl, dtype=torch.float32)

        qa = all_a[qi][gt].to(device)
        qv = all_v[qi][gt].to(device)

        s_a2v = calculate_sim_global(qa, all_cv)
        s_v2a = calculate_sim_global(qv, all_ca)

        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(rel_t, s_a2v.cpu(), k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(rel_t, s_a2v.cpu(), k)
            metrics["recall_v2a"][k] += recall_at_k(rel_t, s_v2a.cpu(), k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(rel_t, s_v2a.cpu(), k)
        metrics["count"] += 1

        del all_cv, all_ca

    return metrics


# ============================================================
# Helpers
# ============================================================

def _empty_metrics(ks):
    return {
        "recall_a2v": {k: 0.0 for k in ks},
        "recall_v2a": {k: 0.0 for k in ks},
        "ndcg_a2v": {k: 0.0 for k in ks},
        "ndcg_v2a": {k: 0.0 for k in ks},
        "count": 0,
    }


def print_metrics(metrics, ks, title=""):
    count = max(metrics["count"], 1)
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    hdr = f"  {'':>6}"
    for k in ks:
        hdr += f" | R@{k:<2}    N@{k:<2}  "
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    row_a2v = f"  {'A2V':>6}"
    for k in ks:
        r = metrics["recall_a2v"][k] / count * 100
        n = metrics["ndcg_a2v"][k] / count * 100
        row_a2v += f" | {r:5.2f}  {n:5.2f} "
    print(row_a2v)

    row_v2a = f"  {'V2A':>6}"
    for k in ks:
        r = metrics["recall_v2a"][k] / count * 100
        n = metrics["ndcg_v2a"][k] / count * 100
        row_v2a += f" | {r:5.2f}  {n:5.2f} "
    print(row_v2a)


def print_latex_row(metrics, ks, name, count):
    vals = []
    for k in ks:
        vals.append(metrics["recall_v2a"][k] / count * 100)
        vals.append(metrics["ndcg_v2a"][k] / count * 100)
    for k in ks:
        vals.append(metrics["recall_a2v"][k] / count * 100)
        vals.append(metrics["ndcg_a2v"][k] / count * 100)
    cols = " & ".join(f"{v:.2f}" for v in vals)
    print(f"  {name} & {cols}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LanguageBind Temporal + Mixed Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--videos_source", default="./data/test_avsync")
    parser.add_argument("--dataset", default="avsync", choices=["avsync", "vggsound"])
    parser.add_argument("--window_size", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--max_segments", type=int, default=20)
    parser.add_argument("--num_distractors", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--video_ckpt", type=str, default="./weights/LanguageBind_Video")
    parser.add_argument("--audio_ckpt", type=str, default="./weights/LanguageBind_Audio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, modality_transform = load_model(
        device=device,
        cache_dir=args.cache_dir,
        video_ckpt=args.video_ckpt,
        audio_ckpt=args.audio_ckpt,
    )

    if args.dataset == "avsync":
        dataset = AvsyncDatasetCrossRetrieval(
            videos_source=args.videos_source,
            window_size=args.window_size,
            stride=args.stride,
            max_segments=args.max_segments,
        )
    else:
        dataset = VGGSOundDatasetCrossRetrieval(
            videos_source=args.videos_source,
            window_size=args.window_size,
            stride=args.stride,
            max_segments=args.max_segments,
        )

    if args.max_videos:
        dataset.video_list = dataset.video_list[:args.max_videos]
    print(f"Dataset: {args.dataset}, Videos: {len(dataset)}")

    ks = args.ks

    all_v, all_a, all_samples = precompute_all_embeddings(
        dataset, model, modality_transform, device, batch_size=args.batch_size
    )

    if len(all_v) == 0:
        raise RuntimeError("No valid embeddings were extracted. Please check paths/codecs/model setup.")

    seg_counts = [e.shape[0] for e in all_v]
    print(f"Segments/video: min={min(seg_counts)}, max={max(seg_counts)}, mean={np.mean(seg_counts):.1f}")
    print(f"Embedding dim: {all_v[0].shape[1]}")

    # Temporal-Only
    t_m = evaluate_temporal_only(all_v, all_a, all_samples, ks)
    print_metrics(t_m, ks, f"Temporal-Only ({np.mean(seg_counts):.0f} cands/query)")

    # Mixed
    results = []
    for M in args.num_distractors:
        aM = min(M, len(all_samples) - 1)
        pool = np.mean(seg_counts) * (aM + 1)
        mm = evaluate_mixed_retrieval(
            all_v, all_a, all_samples,
            num_distractors=aM,
            ks=ks,
            seed=args.seed,
            device=device,
        )
        print_metrics(mm, ks, f"Mixed: {aM + 1} videos (~{pool:.0f} cands/query)")
        results.append((aM + 1, pool, mm))

    # LaTeX
    print(f"\n{'=' * 65}")
    print("  LaTeX-friendly summary")
    print(f"{'=' * 65}")
    ct = max(t_m["count"], 1)
    print_latex_row(t_m, ks, "LanguageBind (temporal)", ct)
    for nv, _, mm in results:
        cm = max(mm["count"], 1)
        print_latex_row(mm, ks, f"LanguageBind (M={nv})", cm)


if __name__ == "__main__":
    main()
