#!/usr/bin/env python
"""
eval_mixed_retrieval.py

Mixed Retrieval Evaluation for FATE.
Tests both semantic discrimination (cross-video) and temporal discrimination (intra-video).

Candidate pool = segments from query video + segments from M distractor videos.
The model must:
  1. Semantically distinguish the correct video from distractors
  2. Temporally locate the aligned segment within the correct video

Usage:
    python -m eval.eval_mixed_retrieval \
        --model_path facebook/pe-av-small \
        --lora_path ./weights/fate-lora \
        --test_dir ./data/avsync15 \
        --dataset avsync \
        --num_distractors 50
"""

import os
import random
import warnings
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchaudio
from torchcodec.decoders import VideoDecoder

from models.pe_av import PeAudioVideoModel, PeAudioVideoProcessor
from peft import PeftModel
from datasets import AvsyncDatasetCrossRetrieval, VGGSOundDatasetCrossRetrieval


# ============================================================
# Input Validation
# ============================================================

def validate_inputs(lora_path, test_dir):
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Evaluation directory does not exist: {test_dir}")

    config_path = os.path.join(lora_path, "adapter_config.json")
    weight_paths = [
        os.path.join(lora_path, name)
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ]
    missing = []
    if not os.path.isfile(config_path):
        missing.append(config_path)
    if not any(os.path.isfile(path) for path in weight_paths):
        missing.append("adapter_model.safetensors (or adapter_model.bin)")
    if missing:
        raise FileNotFoundError(
            "Incomplete FATE checkpoint. Missing: " + ", ".join(missing)
        )


# ============================================================
# Metric Functions
# ============================================================

def recall_at_k(relevance, scores, k):
    """Fraction of relevant items found in top-k."""
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
    """Normalized Discounted Cumulative Gain at k."""
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
    discounts = torch.log2(
        torch.arange(k, device=relevance.device, dtype=torch.float32) + 2.0
    )
    dcg = (gains / discounts).sum()
    ideal_rel = torch.sort(relevance, descending=True).values[:k]
    ideal_gains = torch.pow(2.0, ideal_rel) - 1.0
    ideal_dcg = (ideal_gains / discounts).sum()
    if ideal_dcg <= 0:
        return 0.0
    return (dcg / ideal_dcg).item()


# ============================================================
# Similarity & Alignment
# ============================================================

def calculate_sim_frame_emb(query_emb, value_emb):
    """
    Frame-level similarity: diagonal inner product averaged over time.
    Args:
        query_emb: [T, D]
        value_emb: [N, T, D]
    Returns:
        scores: [N]
    """
    # (N, T, D) @ (D, T) -> (N, T, T)
    scores_matrix = torch.matmul(value_emb, query_emb.transpose(0, 1))
    # Extract diagonal: (N, T)
    diag_scores = scores_matrix.diagonal(dim1=-2, dim2=-1)
    # Average over time: (N,)
    return diag_scores.mean(dim=-1)


def align_temporal_dim(embeds, target_T):
    """
    Align temporal dimension to target_T using nearest interpolation.
    Args:
        embeds: [N, T, D] or [1, T, D]
    Returns:
        [N, target_T, D]
    """
    if embeds.shape[1] == target_T:
        return embeds
    # [N, D, T] for F.interpolate
    embeds_t = embeds.permute(0, 2, 1).float()
    embeds_t = F.interpolate(embeds_t, size=target_T, mode="nearest")
    return embeds_t.permute(0, 2, 1)


# ============================================================
# Embedding Extraction
# ============================================================

def get_embeddings(sample, model, processor, device):
    """
    Extract frame-level embeddings for all segments in one video sample.
    Returns:
        v_emb: [N_seg, T, D]  (normalized)
        a_emb: [N_seg, T, D]  (normalized)
    """
    video_path = sample["video_path"]
    segments = sample["segments"]

    vr = VideoDecoder(video_path)
    fps = max(vr.metadata.average_fps, 1e-3)
    num_frames = int(vr.metadata.num_frames)

    wav, sr = torchaudio.load(video_path)
    if sr != 48000:
        wav = torchaudio.transforms.Resample(sr, 48000)(wav)
        sr = 48000
    wav = wav.mean(dim=0, keepdim=True)  # mono

    clip_videos, clip_audios = [], []
    for s, e in segments:
        s, e = float(s), float(e)

        # --- video ---
        start_f = max(0, int(s * fps))
        end_f = min(max(start_f + 1, int(e * fps)), num_frames)
        try:
            if start_f >= num_frames or end_f <= start_f:
                v_clip = torch.zeros((1, 3, 336, 336), dtype=torch.uint8)
            else:
                indices = np.arange(start_f, end_f)
                v_clip = vr.get_frames_at(indices=indices).data
                if v_clip.shape[0] == 0:
                    v_clip = torch.zeros((1, 3, 336, 336), dtype=torch.uint8)
        except Exception:
            v_clip = torch.zeros((1, 3, 336, 336), dtype=torch.uint8)

        # --- audio ---
        start_a = max(0, int(s * sr))
        end_a = min(max(start_a + 1, int(e * sr)), wav.shape[1])
        if end_a <= start_a:
            a_clip = np.zeros(int(2.0 * sr), dtype=np.float32)
        else:
            a_clip = wav[:, start_a:end_a][0].numpy()

        clip_videos.append(v_clip)
        clip_audios.append(a_clip)

    del vr

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        inputs = processor(
            videos=clip_videos,
            audio=clip_audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=48000,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        v_emb = F.normalize(outputs.video_frame_embeds.float(), dim=-1)
        a_emb = F.normalize(outputs.audio_frame_embeds.float(), dim=-1)

    return v_emb, a_emb


# ============================================================
# Phase 1: Pre-compute All Embeddings
# ============================================================

def precompute_all_embeddings(dataset, model, processor, device):
    """
    Compute embeddings for every video in the dataset.
    Returns lists of CPU tensors and sample dicts.
    """
    all_v_embs = []
    all_a_embs = []
    all_samples = []
    failed = 0

    for idx in tqdm(range(len(dataset)), desc="Pre-computing embeddings"):
        sample = dataset[idx]
        try:
            v_emb, a_emb = get_embeddings(sample, model, processor, device)
            all_v_embs.append(v_emb.cpu())
            all_a_embs.append(a_emb.cpu())
            all_samples.append(sample)
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Warning: {sample['video_path']}: {e}")

    print(f"Embeddings computed: {len(all_samples)} ok, {failed} failed")
    return all_v_embs, all_a_embs, all_samples


# ============================================================
# Phase 2a: Temporal-Only Retrieval (existing benchmark)
# ============================================================

def evaluate_temporal_only(all_v_embs, all_a_embs, all_samples, ks):
    """
    Standard intra-video temporal retrieval.
    Candidate pool = segments from the SAME video only (~17 candidates).
    """
    metrics = _empty_metrics(ks)

    for i in range(len(all_samples)):
        sample = all_samples[i]
        v_emb = all_v_embs[i]
        a_emb = all_a_embs[i]
        relevance = sample["relevance"].float()
        gt_idx = int(sample["gt_index"])

        # A2V: audio query -> video candidates
        scores_a2v = calculate_sim_frame_emb(a_emb[gt_idx], v_emb)
        # V2A: video query -> audio candidates
        scores_v2a = calculate_sim_frame_emb(v_emb[gt_idx], a_emb)

        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(relevance, scores_a2v, k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(relevance, scores_a2v, k)
            metrics["recall_v2a"][k] += recall_at_k(relevance, scores_v2a, k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(relevance, scores_v2a, k)
        metrics["count"] += 1

    return metrics


# ============================================================
# Phase 2b: Mixed Retrieval (NEW)
# ============================================================

def evaluate_mixed_retrieval(
    all_v_embs, all_a_embs, all_samples,
    num_distractors, ks, seed=42, device=None,
):
    """
    Mixed retrieval evaluation.

    For each query video:
      - Audio/Video query = GT segment embedding
      - Candidate pool   = ALL segments from query video
                          + ALL segments from M distractor videos
      - Relevance: same as original for query video segments;
                   0 for all distractor segments.

    This tests semantic discrimination (finding the right video among
    distractors) AND temporal discrimination (finding the right segment
    within the correct video) simultaneously.
    """
    if device is None:
        device = torch.device("cpu")

    rng = random.Random(seed)
    num_videos = len(all_samples)
    all_indices = list(range(num_videos))
    actual_M = min(num_distractors, num_videos - 1)

    metrics = _empty_metrics(ks)

    for query_idx in tqdm(
        range(num_videos),
        desc=f"Mixed retrieval (M={actual_M + 1} videos)",
    ):
        sample = all_samples[query_idx]
        gt_idx = int(sample["gt_index"])
        q_v_emb = all_v_embs[query_idx]   # [N_q, T_q, D]
        q_a_emb = all_a_embs[query_idx]   # [N_q, T_q, D]

        # ---- select distractor videos ----
        others = [i for i in all_indices if i != query_idx]
        distractors = rng.sample(others, actual_M)

        # ---- build candidate pool ----
        cand_v_list = [q_v_emb]
        cand_a_list = [q_a_emb]
        relevance_list = list(sample["relevance"].numpy().astype(np.float32))

        for d_idx in distractors:
            cand_v_list.append(all_v_embs[d_idx])
            cand_a_list.append(all_a_embs[d_idx])
            n_seg = all_v_embs[d_idx].shape[0]
            relevance_list.extend([0.0] * n_seg)

        # ---- align temporal dimensions ----
        all_Ts = [e.shape[1] for e in cand_v_list + cand_a_list]
        target_T = min(all_Ts)

        cand_v_list = [align_temporal_dim(e, target_T) for e in cand_v_list]
        cand_a_list = [align_temporal_dim(e, target_T) for e in cand_a_list]

        all_cand_v = torch.cat(cand_v_list, dim=0).to(device)  # [N_total, T, D]
        all_cand_a = torch.cat(cand_a_list, dim=0).to(device)

        relevance_t = torch.tensor(relevance_list, dtype=torch.float32)

        # ---- query embeddings (aligned) ----
        qa = align_temporal_dim(
            q_a_emb[gt_idx : gt_idx + 1], target_T
        )[0].to(device)
        qv = align_temporal_dim(
            q_v_emb[gt_idx : gt_idx + 1], target_T
        )[0].to(device)

        # ---- A2V: audio query -> all video candidates ----
        scores_a2v = calculate_sim_frame_emb(qa, all_cand_v)
        # ---- V2A: video query -> all audio candidates ----
        scores_v2a = calculate_sim_frame_emb(qv, all_cand_a)

        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(relevance_t, scores_a2v.cpu(), k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(relevance_t, scores_a2v.cpu(), k)
            metrics["recall_v2a"][k] += recall_at_k(relevance_t, scores_v2a.cpu(), k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(relevance_t, scores_v2a.cpu(), k)
        metrics["count"] += 1

        # free GPU tensors
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

    # ---- Table header ----
    hdr = f"  {'':>6}"
    for k in ks:
        hdr += f" | R@{k:<2}    N@{k:<2}  "
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    # ---- A2V row ----
    row_a2v = f"  {'A2V':>6}"
    for k in ks:
        r = metrics["recall_a2v"][k] / count * 100
        n = metrics["ndcg_a2v"][k] / count * 100
        row_a2v += f" | {r:5.2f}  {n:5.2f} "
    print(row_a2v)

    # ---- V2A row ----
    row_v2a = f"  {'V2A':>6}"
    for k in ks:
        r = metrics["recall_v2a"][k] / count * 100
        n = metrics["ndcg_v2a"][k] / count * 100
        row_v2a += f" | {r:5.2f}  {n:5.2f} "
    print(row_v2a)


def print_latex_row(metrics, ks, method_name, count):
    """Print a LaTeX table row for easy copy-paste."""
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
        description="Mixed Retrieval Evaluation for FATE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str,
        default="facebook/pe-av-small",
        help="Local path or Hugging Face ID for the base PE-AV model",
    )
    parser.add_argument(
        "--lora_path", type=str, default="./weights/fate-lora",
        help="Local path to the released FATE LoRA checkpoint",
    )
    parser.add_argument(
        "--test_dir", type=str,
        default="./data/test_avsync",
        help="Directory containing test videos",
    )
    parser.add_argument(
        "--dataset", type=str, default="avsync",
        choices=["avsync", "vggsound"],
        help="Dataset type",
    )
    parser.add_argument(
        "--num_distractors", type=int, nargs="+",
        default=[10, 50, 100],
        help="Number of distractor videos (multiple values for scaling experiment)",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[1, 3, 5],
        help="Values of k for R@k and NDCG@k",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for distractor selection",
    )
    parser.add_argument(
        "--window_size", type=float, default=2.0,
        help="Segment window size in seconds",
    )
    parser.add_argument(
        "--stride", type=float, default=0.5,
        help="Segment stride in seconds",
    )
    parser.add_argument(
        "--max_segments", type=int, default=20,
        help="Max segments per video",
    )
    parser.add_argument(
        "--max_videos", type=int, default=None,
        help="Optionally evaluate only the first N videos (useful for a smoke test)",
    )
    args = parser.parse_args()
    validate_inputs(args.lora_path, args.test_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    print(f"Loading model from {args.model_path} ...")
    model = PeAudioVideoModel.from_pretrained(args.model_path)
    processor = PeAudioVideoProcessor.from_pretrained(args.model_path)

    print(f"Loading FATE LoRA from {args.lora_path} ...")
    model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)

    model = model.to(device).eval()

    # ---- Load dataset ----
    if args.dataset == "avsync":
        dataset = AvsyncDatasetCrossRetrieval(
            videos_source=args.test_dir,
            window_size=args.window_size,
            stride=args.stride,
            max_segments=args.max_segments,
        )
    else:
        dataset = VGGSOundDatasetCrossRetrieval(
            videos_source=args.test_dir,
            window_size=args.window_size,
            stride=args.stride,
            max_segments=args.max_segments,
        )
    if args.max_videos is not None:
        if args.max_videos < 1:
            raise ValueError("--max_videos must be at least 1")
        dataset.video_list = dataset.video_list[: args.max_videos]
    if len(dataset) == 0:
        raise RuntimeError(f"No evaluation videos found under {args.test_dir}")
    print(f"Dataset: {args.dataset}, Total videos: {len(dataset)}")

    # ---- Phase 1: Pre-compute all embeddings ----
    all_v_embs, all_a_embs, all_samples = precompute_all_embeddings(
        dataset, model, processor, device
    )

    # Report statistics
    seg_counts = [e.shape[0] for e in all_v_embs]
    T_dims = [e.shape[1] for e in all_v_embs]
    print(f"Segments per video: min={min(seg_counts)}, max={max(seg_counts)}, "
          f"mean={np.mean(seg_counts):.1f}")
    print(f"Temporal dim T: min={min(T_dims)}, max={max(T_dims)}, "
          f"mean={np.mean(T_dims):.1f}")

    ks = args.ks

    # ---- Phase 2a: Temporal-Only retrieval (for comparison) ----
    t_metrics = evaluate_temporal_only(all_v_embs, all_a_embs, all_samples, ks)
    avg_pool_temporal = np.mean(seg_counts)
    print_metrics(
        t_metrics, ks,
        f"Temporal-Only Retrieval (avg {avg_pool_temporal:.0f} candidates/query)",
    )

    # ---- Phase 2b: Mixed retrieval at different pool sizes ----
    results_summary = []
    for M in args.num_distractors:
        actual_M = min(M, len(all_samples) - 1)
        avg_pool = np.mean(seg_counts) * (actual_M + 1)

        m_metrics = evaluate_mixed_retrieval(
            all_v_embs, all_a_embs, all_samples,
            num_distractors=actual_M,
            ks=ks,
            seed=args.seed,
            device=device,
        )
        title = (
            f"Mixed Retrieval: {actual_M + 1} videos "
            f"(~{avg_pool:.0f} candidates/query)"
        )
        print_metrics(m_metrics, ks, title)
        results_summary.append((actual_M + 1, avg_pool, m_metrics))

    # ---- Summary: LaTeX-friendly output ----
    print(f"\n{'=' * 65}")
    print("  LaTeX-friendly summary (copy-paste into your table)")
    print(f"{'=' * 65}")
    count_t = max(t_metrics["count"], 1)
    print(f"  % Temporal-Only ({avg_pool_temporal:.0f} cands)")
    print_latex_row(t_metrics, ks, "FATE (temporal)", count_t)

    for num_vids, avg_pool, m_metrics in results_summary:
        count_m = max(m_metrics["count"], 1)
        print(f"  % Mixed M={num_vids} (~{avg_pool:.0f} cands)")
        print_latex_row(m_metrics, ks, f"FATE (M={num_vids})", count_m)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message=r"PeAudioVideoModel does not expose.*"
    )
    main()