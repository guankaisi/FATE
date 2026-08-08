#!/usr/bin/env python
"""
eval_cavp_retrieval.py

CAVP evaluation for both Temporal-Only and Mixed Retrieval.
Uses global embeddings for similarity computation.

Usage:
    python eval_cavp_retrieval.py \
        --dataset_path /data1/kaisi/datasets/test_avsync \
        --dataset avsync \
        --num_distractors 50 \
        --max_videos 150
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import librosa
import cv2
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import subprocess
import warnings
import traceback

warnings.filterwarnings("ignore")

# Add paths
diff_foley_path = os.path.join(os.path.dirname(__file__), 'Diff-Foley')
sys.path.insert(0, diff_foley_path)
sys.path.insert(0, os.path.join(diff_foley_path, 'inference'))

from demo_util import instantiate_from_config, reencode_video_with_diff_fps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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


def calculate_sim_global(query_emb, value_emb):
    return torch.matmul(value_emb, query_emb)


# ============================================================
# CAVP Model Loading
# ============================================================

def which_ffmpeg():
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.decode('utf-8').replace('\n', '')


def load_cavp_model(config_path, ckpt_path, device):
    print(f"Loading CAVP model from {ckpt_path}")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd.get("state_dict", pl_sd)

    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # Fix BatchNorm mismatch
    bn_prefix = None
    for key in new_sd:
        if 'bn.running_mean' in key and 'spec_encoder' in key:
            bn_prefix = key.replace('.running_mean', '')
            break

    if bn_prefix:
        new_sd = {k: v for k, v in new_sd.items() if not k.startswith(bn_prefix)}

    model.spec_encoder.bn = nn.BatchNorm2d(80)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Loaded CAVP: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to(device).eval()
    return model


# ============================================================
# CAVP Feature Extraction
# ============================================================

def extract_audio_from_video(video_path, start_second, duration, tmp_path, output_audio_path):
    os.makedirs(tmp_path, exist_ok=True)
    cmd = (f'{which_ffmpeg()} -hide_banner -loglevel panic '
           f'-y -ss {start_second} -t {duration} -i {video_path} '
           f'-vn -acodec pcm_s16le -ar 22050 -ac 1 {output_audio_path}')
    try:
        ret = subprocess.call(cmd.split())
        if ret != 0 or not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
            return None
        return output_audio_path
    except Exception:
        return None


def get_spectrogram(audio_path, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    try:
        wav, _ = librosa.load(audio_path, sr=sr)
    except Exception:
        return None
    wav = wav.reshape(-1)
    if wav.size == 0:
        return None
    if wav.size < n_fft:
        wav = np.pad(wav, (0, n_fft - wav.size), mode='constant')
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=125, fmax=7600, power=1, center=False
        )
    except Exception:
        return None
    mel_spec = np.log10(np.maximum(mel_spec, 1e-5))
    mel_spec = (mel_spec * 20 + 20) / 100
    mel_spec = np.clip(mel_spec, 0, 1.0)
    return mel_spec


def extract_video_feat(model, video_path, start_sec, duration, device,
                       fps=4, batch_size=40, tmp_path="./tmp_folder"):
    os.makedirs(tmp_path, exist_ok=True)
    try:
        video_path_low_fps = reencode_video_with_diff_fps(
            video_path, tmp_path, fps, start_sec, duration
        )
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path_low_fps)
    if not cap.isOpened():
        return None

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    feat_batch_list = []
    video_feats = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_tensor = img_transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
            feat_batch_list.append(rgb_tensor)
        except Exception:
            continue

        if len(feat_batch_list) == batch_size:
            input_feats = torch.cat(feat_batch_list, 0).unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    vf = model.encode_video(input_feats, normalize=False, pool=False)
                    if vf.shape[1] < 16:
                        vf = vf.mean(dim=1)
                    else:
                        vf = model.video_pool(vf.permute(0, 2, 1)).squeeze(2)
                    vf = F.normalize(vf, dim=-1)
                video_feats.append(vf.cpu().numpy())
            except Exception:
                pass
            feat_batch_list = []

    if len(feat_batch_list) > 0:
        input_feats = torch.cat(feat_batch_list, 0).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                vf = model.encode_video(input_feats, normalize=False, pool=False)
                if vf.shape[1] < 16:
                    vf = vf.mean(dim=1)
                else:
                    vf = model.video_pool(vf.permute(0, 2, 1)).squeeze(2)
                vf = F.normalize(vf, dim=-1)
            video_feats.append(vf.cpu().numpy())
        except Exception:
            pass

    cap.release()
    if len(video_feats) == 0:
        return None
    return np.mean(np.concatenate(video_feats, axis=0), axis=0)


def extract_audio_feat(model, video_path, start_sec, duration, device, tmp_path="./tmp_folder"):
    os.makedirs(tmp_path, exist_ok=True)
    audio_path = os.path.join(tmp_path, f"tmp_audio_{os.getpid()}_{start_sec:.2f}.wav")
    try:
        actual_path = extract_audio_from_video(video_path, start_sec, duration, tmp_path, audio_path)
        if actual_path is None:
            return None
        mel_spec = get_spectrogram(actual_path)
        if mel_spec is None:
            return None
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            sf = model.encode_spec(mel_tensor, normalize=False, pool=False)
            if sf.shape[1] < 16:
                sf = sf.mean(dim=1)
            else:
                sf = model.spec_pool(sf.permute(0, 2, 1)).squeeze(2)
            sf = F.normalize(sf, dim=-1)
        return sf.cpu().numpy().squeeze()
    except Exception:
        return None
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ============================================================
# Embedding Extraction (per video)
# ============================================================

def get_embeddings(sample, model, device, fps=4, batch_size=40, tmp_path="./tmp_folder"):
    """
    Extract CAVP global embeddings for all segments of one video.
    Returns:
        v_emb: [N_seg, D] tensor, normalized
        a_emb: [N_seg, D] tensor, normalized
    """
    video_path = sample["video_path"]
    segments = sample["segments"]

    v_feats = []
    a_feats = []

    for seg_start, seg_end in segments:
        seg_start, seg_end = float(seg_start), float(seg_end)
        duration = seg_end - seg_start

        vf = extract_video_feat(model, video_path, seg_start, duration,
                                device, fps=fps, batch_size=batch_size, tmp_path=tmp_path)
        af = extract_audio_feat(model, video_path, seg_start, duration,
                                device, tmp_path=tmp_path)

        if vf is None:
            vf = np.zeros(512, dtype=np.float32)
        if af is None:
            af = np.zeros(512, dtype=np.float32)

        v_feats.append(vf)
        a_feats.append(af)

    v_emb = F.normalize(torch.from_numpy(np.stack(v_feats)).float(), dim=-1)
    a_emb = F.normalize(torch.from_numpy(np.stack(a_feats)).float(), dim=-1)

    return v_emb, a_emb


# ============================================================
# Phase 1: Pre-compute All Embeddings
# ============================================================

def precompute_all_embeddings(dataset, model, device, fps=4, batch_size=40,
                              tmp_path="./tmp_folder"):
    all_v_embs = []
    all_a_embs = []
    all_samples = []
    failed = 0

    for idx in tqdm(range(len(dataset)), desc="Pre-computing CAVP embeddings"):
        sample = dataset[idx]
        try:
            v_emb, a_emb = get_embeddings(
                sample, model, device, fps=fps,
                batch_size=batch_size, tmp_path=tmp_path
            )
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
        v_emb = all_v_embs[i]
        a_emb = all_a_embs[i]
        relevance = sample["relevance"].float()
        gt_idx = int(sample["gt_index"])

        scores_a2v = calculate_sim_global(a_emb[gt_idx], v_emb)
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
        q_v_emb = all_v_embs[query_idx]
        q_a_emb = all_a_embs[query_idx]

        others = [i for i in all_indices if i != query_idx]
        distractors = rng.sample(others, actual_M)

        cand_v_list = [q_v_emb]
        cand_a_list = [q_a_emb]
        relevance_list = list(sample["relevance"].numpy().astype(np.float32))

        for d_idx in distractors:
            cand_v_list.append(all_v_embs[d_idx])
            cand_a_list.append(all_a_embs[d_idx])
            n_seg = all_v_embs[d_idx].shape[0]
            relevance_list.extend([0.0] * n_seg)

        all_cand_v = torch.cat(cand_v_list, dim=0).to(device)
        all_cand_a = torch.cat(cand_a_list, dim=0).to(device)
        relevance_t = torch.tensor(relevance_list, dtype=torch.float32)

        qa = q_a_emb[gt_idx].to(device)
        qv = q_v_emb[gt_idx].to(device)

        scores_a2v = calculate_sim_global(qa, all_cand_v)
        scores_v2a = calculate_sim_global(qv, all_cand_a)

        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(relevance_t, scores_a2v.cpu(), k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(relevance_t, scores_a2v.cpu(), k)
            metrics["recall_v2a"][k] += recall_at_k(relevance_t, scores_v2a.cpu(), k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(relevance_t, scores_v2a.cpu(), k)
        metrics["count"] += 1

        del all_cand_v, all_cand_a

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
        description="CAVP Temporal + Mixed Retrieval Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_path", type=str,
                        default="Diff-Foley/inference/config/Stage1_CAVP.yaml")
    parser.add_argument("--ckpt_path", type=str,
                        default="./weights/cavp/cavp_epoch66.ckpt")
    parser.add_argument("--dataset_path", type=str,
                        default="./data/test_avsync")
    parser.add_argument("--dataset", type=str, default="avsync",
                        choices=["avsync", "vggsound"])
    parser.add_argument("--window_size", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--max_segments", type=int, default=20)
    parser.add_argument("--num_distractors", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--tmp_path", type=str, default="./tmp_cavp_eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    config_path = os.path.join(os.path.dirname(__file__), args.config_path)
    model = load_cavp_model(config_path, args.ckpt_path, device)

    # Load dataset
    if args.dataset == "avsync":
        dataset = AvsyncDatasetCrossRetrieval(
            videos_source=args.dataset_path,
            window_size=args.window_size,
            stride=args.stride,
            max_segments=args.max_segments,
        )
    else:
        dataset = VGGSOundDatasetCrossRetrieval(
            videos_source=args.dataset_path,
            window_size=args.window_size,
            stride=args.stride,
            max_segments=args.max_segments,
        )
    print(f"Dataset: {args.dataset}, Total videos: {len(dataset)}")

    ks = args.ks

    # Phase 1: Pre-compute all embeddings
    all_v_embs, all_a_embs, all_samples = precompute_all_embeddings(
        dataset, model, device,
        fps=args.fps, batch_size=args.batch_size, tmp_path=args.tmp_path
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
    print_latex_row(t_metrics, ks, "CAVP (temporal)", count_t)

    for num_vids, avg_pool_m, m_metrics in results_summary:
        count_m = max(m_metrics["count"], 1)
        print(f"  % Mixed M={num_vids} (~{avg_pool_m:.0f} cands)")
        print_latex_row(m_metrics, ks, f"CAVP (M={num_vids})", count_m)


if __name__ == "__main__":
    main()