#!/usr/bin/env python
"""
eval_imagebind_retrieval.py

ImageBind evaluation for both Temporal-Only and Mixed Retrieval.
Uses global embeddings (CLS) for similarity computation.

Usage:
    python eval_imagebind_retrieval.py \
        --videos_source /data1/kaisi/datasets/test_avsync \
        --num_distractors 50 \
        --max_videos 150
"""

import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import av
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset
from torchvision import transforms
from pytorchvideo import transforms as pv_transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.io import read_video
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_BASELINES = os.environ.get("FATE_BASELINE_ROOT", "/data1/kaisi/sync/baselines")
BASELINES_LIB_ROOT = ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "baselines", "imagebind")) else os.path.dirname(EXTERNAL_BASELINES)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(BASELINES_LIB_ROOT, "baselines"))

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.data import waveform2melspec, SpatialCrop

from datasets import AvsyncDatasetCrossRetrieval, VGGSOundDatasetCrossRetrieval


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


# ============================================================
# ImageBind Video/Audio Loading
# ============================================================

def load_video_segment(video_path, start_time, duration):
    """Load and preprocess a video segment for ImageBind."""
    frames = []
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)

        timestamp = int(start_time / stream.time_base)
        container.seek(timestamp, stream=stream)

        max_frames = int(duration * fps) + 10

        for frame in container.decode(video=0):
            if frame.time < start_time:
                continue
            if frame.time > start_time + duration:
                break
            frames.append(frame.to_rgb().to_ndarray())
            if len(frames) >= max_frames:
                break
        container.close()
    except Exception:
        return torch.zeros(3, 3, 2, 224, 224)

    if not frames:
        return torch.zeros(3, 3, 2, 224, 224)

    frames = torch.as_tensor(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0

    transform = transforms.Compose([
        pv_transforms.ShortSideScale(224),
        NormalizeVideo(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    frames = transform(frames)

    temporal_subsample = pv_transforms.UniformTemporalSubsample(num_samples=2)
    frames = temporal_subsample(frames)

    spatial_crop = SpatialCrop(224, num_crops=3)
    crops = spatial_crop([frames])

    return torch.stack(crops, dim=0)  # [3, C, T, H, W]


def load_audio_segment(video_path, start_time, duration):
    """Load and preprocess an audio segment for ImageBind."""
    try:
        waveform, sr = torchaudio.load(video_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            sr = 16000

        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)

        if start_sample >= waveform.shape[1]:
            return torch.zeros(3, 1, 128, 204)

        waveform_clip = waveform[:, start_sample:end_sample]

        if waveform_clip.shape[1] == 0:
            return torch.zeros(3, 1, 128, 204)

        melspec = waveform2melspec(waveform_clip, sr, 128, 204)

        mean = -4.268
        std = 9.138
        normalize = transforms.Normalize(mean=mean, std=std)
        melspec = normalize(melspec)

        return melspec.unsqueeze(0).repeat(3, 1, 1, 1)  # [3, 1, 128, 204]
    except Exception:
        return torch.zeros(3, 1, 128, 204)


# ============================================================
# Embedding Extraction
# ============================================================

def get_embeddings(sample, model, device, batch_size=4):
    """
    Extract ImageBind global embeddings for all segments of one video.
    Returns:
        v_emb: [N_seg, D] normalized
        a_emb: [N_seg, D] normalized
    """
    video_path = sample["video_path"]
    segments = sample["segments"]

    all_v_embs = []
    all_a_embs = []

    with torch.no_grad():
        for start in range(0, len(segments), batch_size):
            batch_segs = segments[start:start + batch_size]
            video_inputs = []
            audio_inputs = []

            for seg_start, seg_end in batch_segs:
                seg_start, seg_end = float(seg_start), float(seg_end)
                duration = seg_end - seg_start

                v = load_video_segment(video_path, seg_start, duration)
                a = load_audio_segment(video_path, seg_start, duration)
                video_inputs.append(v)
                audio_inputs.append(a)

            if len(video_inputs) == 0:
                continue

            video_batch = torch.stack(video_inputs, dim=0).to(device)
            audio_batch = torch.stack(audio_inputs, dim=0).to(device)

            inputs = {
                ModalityType.VISION: video_batch,
                ModalityType.AUDIO: audio_batch,
            }

            outputs = model(inputs)

            v_emb = outputs[ModalityType.VISION]  # [B, D]
            a_emb = outputs[ModalityType.AUDIO]   # [B, D]

            all_v_embs.append(v_emb.cpu())
            all_a_embs.append(a_emb.cpu())

    if len(all_v_embs) == 0:
        return None, None

    v_emb = F.normalize(torch.cat(all_v_embs, dim=0), dim=-1)
    a_emb = F.normalize(torch.cat(all_a_embs, dim=0), dim=-1)

    return v_emb, a_emb


# ============================================================
# Similarity (Global)
# ============================================================

def calculate_sim_global(query_emb, value_emb):
    """
    Global cosine similarity.
    Args:
        query_emb: [D]
        value_emb: [N, D]
    Returns:
        scores: [N]
    """
    return torch.matmul(value_emb, query_emb)


# ============================================================
# Phase 1: Pre-compute All Embeddings
# ============================================================

def precompute_all_embeddings(dataset, model, device, batch_size=4):
    all_v_embs = []
    all_a_embs = []
    all_samples = []
    failed = 0

    for idx in tqdm(range(len(dataset)), desc="Pre-computing ImageBind embeddings"):
        sample = dataset[idx]
        try:
            v_emb, a_emb = get_embeddings(sample, model, device, batch_size)
            if v_emb is None or a_emb is None:
                failed += 1
                continue
            all_v_embs.append(v_emb)
            all_a_embs.append(a_emb)
            all_samples.append(sample)
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Warning: {sample['video_path']}: {e}")

    print(f"Embeddings computed: {len(all_samples)} ok, {failed} failed")
    return all_v_embs, all_a_embs, all_samples


# ============================================================
# Phase 2a: Temporal-Only Retrieval
# ============================================================

def evaluate_temporal_only(all_v_embs, all_a_embs, all_samples, ks):
    metrics = _empty_metrics(ks)

    for i in range(len(all_samples)):
        sample = all_samples[i]
        v_emb = all_v_embs[i]  # [N_seg, D]
        a_emb = all_a_embs[i]  # [N_seg, D]
        relevance = sample["relevance"].float()
        gt_idx = int(sample["gt_index"])

        # A2V
        scores_a2v = calculate_sim_global(a_emb[gt_idx], v_emb)
        # V2A
        scores_v2a = calculate_sim_global(v_emb[gt_idx], a_emb)

        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(relevance, scores_a2v, k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(relevance, scores_a2v, k)
            metrics["recall_v2a"][k] += recall_at_k(relevance, scores_v2a, k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(relevance, scores_v2a, k)
        metrics["count"] += 1

    return metrics


# ============================================================
# Phase 2b: Mixed Retrieval
# ============================================================

def evaluate_mixed_retrieval(all_v_embs, all_a_embs, all_samples,
                             num_distractors, ks, seed=42, device=None):
    if device is None:
        device = torch.device("cpu")

    rng = random.Random(seed)
    num_videos = len(all_samples)
    all_indices = list(range(num_videos))
    actual_M = min(num_distractors, num_videos - 1)

    metrics = _empty_metrics(ks)

    for query_idx in tqdm(range(num_videos),
                          desc=f"Mixed retrieval (M={actual_M + 1} videos)"):
        sample = all_samples[query_idx]
        gt_idx = int(sample["gt_index"])
        q_v_emb = all_v_embs[query_idx]  # [N_q, D]
        q_a_emb = all_a_embs[query_idx]  # [N_q, D]

        # Select distractors
        others = [i for i in all_indices if i != query_idx]
        distractors = rng.sample(others, actual_M)

        # Build candidate pool
        cand_v_list = [q_v_emb]
        cand_a_list = [q_a_emb]
        relevance_list = list(sample["relevance"].numpy().astype(np.float32))

        for d_idx in distractors:
            cand_v_list.append(all_v_embs[d_idx])
            cand_a_list.append(all_a_embs[d_idx])
            n_seg = all_v_embs[d_idx].shape[0]
            relevance_list.extend([0.0] * n_seg)

        # Global embeddings: no temporal alignment needed
        all_cand_v = torch.cat(cand_v_list, dim=0).to(device)  # [N_total, D]
        all_cand_a = torch.cat(cand_a_list, dim=0).to(device)  # [N_total, D]
        relevance_t = torch.tensor(relevance_list, dtype=torch.float32)

        qa = q_a_emb[gt_idx].to(device)  # [D]
        qv = q_v_emb[gt_idx].to(device)  # [D]

        scores_a2v = calculate_sim_global(qa, all_cand_v)
        scores_v2a = calculate_sim_global(qv, all_cand_a)

        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(relevance_t, scores_a2v.cpu(), k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(relevance_t, scores_a2v.cpu(), k)
            metrics["recall_v2a"][k] += recall_at_k(relevance_t, scores_v2a.cpu(), k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(relevance_t, scores_v2a.cpu(), k)
        metrics["count"] += 1

        del all_cand_v, all_cand_a, scores_a2v, scores_v2a

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


def print_latex_row(metrics, ks, method_name, count):
    vals = []
    for k in ks:
        vals.append(metrics["recall_v2a"][k] / count * 100)
        vals.append(metrics["ndcg_v2a"][k] / count * 100)
    for k in ks:
        vals.append(metrics["recall_a2v"][k] / count * 100)
        vals.append(metrics["ndcg_a2v"][k] / count * 100)
    cols = " & ".join(f"{v:.2f}" for v in vals)
    print(f"  {method_name} & {cols} \\\\")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ImageBind Temporal + Mixed Retrieval Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--videos_source", default="./data/test_avsync")
    parser.add_argument("--dataset", type=str, default="avsync",
                        choices=["avsync", "vggsound"])
    parser.add_argument("--window_size", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--max_segments", type=int, default=20)
    parser.add_argument("--num_distractors", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for ImageBind inference per video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading ImageBind model...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Load dataset (use AvsyncDatasetCrossRetrieval for consistent evaluation)
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
    print(f"Dataset: {args.dataset}, Total videos: {len(dataset)}")

    ks = args.ks

    # Phase 1: Pre-compute all embeddings
    all_v_embs, all_a_embs, all_samples = precompute_all_embeddings(
        dataset, model, device, batch_size=args.batch_size
    )

    seg_counts = [e.shape[0] for e in all_v_embs]
    emb_dim = all_v_embs[0].shape[1]
    print(f"Segments per video: min={min(seg_counts)}, max={max(seg_counts)}, "
          f"mean={np.mean(seg_counts):.1f}")
    print(f"Embedding dim D: {emb_dim}")

    # Phase 2a: Temporal-Only
    t_metrics = evaluate_temporal_only(all_v_embs, all_a_embs, all_samples, ks)
    avg_pool = np.mean(seg_counts)
    print_metrics(t_metrics, ks,
                  f"Temporal-Only Retrieval (avg {avg_pool:.0f} candidates/query)")

    # Phase 2b: Mixed Retrieval
    results_summary = []
    for M in args.num_distractors:
        actual_M = min(M, len(all_samples) - 1)
        avg_pool_m = np.mean(seg_counts) * (actual_M + 1)

        m_metrics = evaluate_mixed_retrieval(
            all_v_embs, all_a_embs, all_samples,
            num_distractors=actual_M, ks=ks,
            seed=args.seed, device=device,
        )
        title = (f"Mixed Retrieval: {actual_M + 1} videos "
                 f"(~{avg_pool_m:.0f} candidates/query)")
        print_metrics(m_metrics, ks, title)
        results_summary.append((actual_M + 1, avg_pool_m, m_metrics))

    # LaTeX output
    print(f"\n{'=' * 65}")
    print("  LaTeX-friendly summary (copy-paste into your table)")
    print(f"{'=' * 65}")
    count_t = max(t_metrics["count"], 1)
    print(f"  % Temporal-Only ({avg_pool:.0f} cands)")
    print_latex_row(t_metrics, ks, "ImageBind (temporal)", count_t)

    for num_vids, avg_pool_m, m_metrics in results_summary:
        count_m = max(m_metrics["count"], 1)
        print(f"  % Mixed M={num_vids} (~{avg_pool_m:.0f} cands)")
        print_latex_row(m_metrics, ks, f"ImageBind (M={num_vids})", count_m)


if __name__ == "__main__":
    main()