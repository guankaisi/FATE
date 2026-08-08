#!/usr/bin/env python
"""
eval_peavs_retrieval.py

PEAVS retrieval evaluation for both Temporal-Only and Mixed Retrieval.
It uses the pretrained PEAVS SyncMetric score as cross-modal similarity.

Usage:
    python eval_peavs_retrieval.py \
        --videos_source /data1/kaisi/datasets/test_avsync \
        --dataset avsync \
        --num_distractors 50 \
        --max_videos 150
"""

import os
import sys
import random
import argparse
import tempfile
import subprocess
import warnings
import importlib.util
import types
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Keep local project modules ahead of site-packages.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_BASELINES = os.environ.get("FATE_BASELINE_ROOT", "/data1/kaisi/sync/baselines")
BASELINES_LIB_ROOT = ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "baselines", "avgen-eval-toolkit")) else os.path.dirname(EXTERNAL_BASELINES)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(BASELINES_LIB_ROOT, "baselines"))

from datasets import AvsyncDatasetCrossRetrieval, VGGSOundDatasetCrossRetrieval

PEAVS_ROOT = os.environ.get(
    "PEAVS_ROOT",
    os.path.join(BASELINES_LIB_ROOT, "baselines", "avgen-eval-toolkit", "PEAVS"),
)
PEAVS_FEAT_ROOT = os.path.join(PEAVS_ROOT, "av_features_extraction")
PEAVS_FEAT_UTILS_DIR = os.path.join(PEAVS_FEAT_ROOT, "utils")
sys.path.insert(0, PEAVS_ROOT)
sys.path.insert(0, PEAVS_FEAT_ROOT)

def _register_peavs_feature_utils():
    """Force `utils.utils` and `utils.io` to resolve to av_features_extraction/utils."""
    for key in list(sys.modules.keys()):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]

    pkg = types.ModuleType("utils")
    pkg.__path__ = [PEAVS_FEAT_UTILS_DIR]
    sys.modules["utils"] = pkg

    for sub_name in ("utils", "io"):
        mod_name = f"utils.{sub_name}"
        mod_path = os.path.join(PEAVS_FEAT_UTILS_DIR, f"{sub_name}.py")
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {mod_name} from {mod_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)


_register_peavs_feature_utils()


@contextmanager
def _pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

from net.modelWrapper import ModelWrapper
from models.i3d.extract_i3d import ExtractI3D
from models.vggish.extract_vggish import ExtractVGGish


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
# PEAVS Loading
# ============================================================

def build_peavs_args(device):
    return SimpleNamespace(
        device=device,
        seed=0,
        batch_size=64,
        output_dim=1,
        attn_dropout=0.1,
        relu_dropout=0.1,
        embed_dropout=0.25,
        res_dropout=0.1,
        out_dropout=0.2,
        layers=3,
        num_heads=8,
        attn_mask=True,
    )


def load_peavs(device, flow_type="raft"):
    print("Loading PEAVS SyncMetric...")
    args = build_peavs_args(device)
    wrapper = ModelWrapper(args)
    wrapper.init_model()
    wrapper.load_model(os.path.join(PEAVS_ROOT, "metric_weights"))
    wrapper.set_eval()

    i3d_args = SimpleNamespace(
        feature_type="i3d",
        stack_size=24,
        step_size=24,
        streams=None,
        flow_type=flow_type,
        extraction_fps=None,
        device=device,
        on_extraction="save_numpy",
        output_path="./unused_output",
        tmp_path="./unused_tmp",
        keep_tmp_files=False,
        show_pred=False,
        config=None,
        video_paths=None,
        file_with_video_paths=None,
    )
    vggish_args = SimpleNamespace(
        feature_type="vggish",
        device=device,
        on_extraction="save_numpy",
        output_path="./unused_output",
        tmp_path="./unused_tmp",
        keep_tmp_files=False,
        show_pred=False,
        config=None,
        video_paths=None,
        file_with_video_paths=None,
    )

    # PEAVS feature extractors use relative checkpoint paths internally.
    with _pushd(PEAVS_FEAT_ROOT):
        i3d_extractor = ExtractI3D(i3d_args)
        vggish_extractor = ExtractVGGish(vggish_args)

    return wrapper, i3d_extractor, vggish_extractor


# ============================================================
# Segment Feature Extraction
# ============================================================

def ffmpeg_cut(video_path, start_sec, end_sec, out_path):
    dur = max(0.05, float(end_sec) - float(start_sec))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{float(start_sec):.6f}",
        "-t", f"{dur:.6f}",
        "-i", video_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        out_path,
    ]
    try:
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return (ret.returncode == 0) and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def extract_segment_features(video_path, seg_start, seg_end, i3d_extractor, vggish_extractor):
    with tempfile.TemporaryDirectory(prefix="peavs_seg_") as td:
        clip_path = os.path.join(td, "clip.mp4")
        ok = ffmpeg_cut(video_path, seg_start, seg_end, clip_path)
        if not ok:
            return None, None

        with _pushd(PEAVS_FEAT_ROOT):
            try:
                a_dict = vggish_extractor.extract(clip_path)
                a_feat = np.asarray(a_dict["vggish"], dtype=np.float32)
            except Exception:
                a_feat = None

            try:
                v_dict = i3d_extractor.extract(clip_path)
                rgb = np.asarray(v_dict["rgb"], dtype=np.float32)
                flow = np.asarray(v_dict["flow"], dtype=np.float32)
                if rgb.shape != flow.shape:
                    v_feat = None
                else:
                    v_feat = rgb + flow
            except Exception:
                v_feat = None

        if a_feat is None or v_feat is None:
            return None, None
        if a_feat.ndim != 2 or v_feat.ndim != 2:
            return None, None
        if a_feat.shape[0] == 0 or v_feat.shape[0] == 0:
            return None, None
        if a_feat.shape[1] != 128 or v_feat.shape[1] != 1024:
            return None, None

        return a_feat, v_feat


def score_pair(model_wrapper, a_feat, v_feat, device):
    xa = torch.from_numpy(a_feat).unsqueeze(0).to(device=device, dtype=torch.float32)
    xv = torch.from_numpy(v_feat).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask_a = torch.ones((1, xa.shape[1]), device=device, dtype=torch.float32)
    mask_v = torch.ones((1, xv.shape[1]), device=device, dtype=torch.float32)

    pred = model_wrapper.feed_forward(xa, xv, aud_mask=mask_a, vid_mask=mask_v)
    pred = torch.clamp(pred * 4.0 + 1.0, min=1.0, max=5.0)
    return float(pred.squeeze().item())


def get_scores_for_video(sample, model_wrapper, i3d_extractor, vggish_extractor, device):
    video_path = sample["video_path"]
    segments = sample["segments"]

    all_a = []
    all_v = []
    valid_mask = []

    for seg_start, seg_end in segments:
        a_feat, v_feat = extract_segment_features(video_path, seg_start, seg_end, i3d_extractor, vggish_extractor)
        if a_feat is None or v_feat is None:
            all_a.append(None)
            all_v.append(None)
            valid_mask.append(False)
        else:
            all_a.append(a_feat)
            all_v.append(v_feat)
            valid_mask.append(True)

    gt = int(sample["gt_index"])
    if gt < 0 or gt >= len(segments) or (not valid_mask[gt]):
        return None, None

    q_a = all_a[gt]
    q_v = all_v[gt]

    s_a2v = []
    s_v2a = []
    for i in range(len(segments)):
        if valid_mask[i]:
            s_a2v.append(score_pair(model_wrapper, q_a, all_v[i], device))
            s_v2a.append(score_pair(model_wrapper, all_a[i], q_v, device))
        else:
            s_a2v.append(-1e9)
            s_v2a.append(-1e9)

    return torch.tensor(s_a2v, dtype=torch.float32), torch.tensor(s_v2a, dtype=torch.float32)


# ============================================================
# Phase 1: Pre-compute
# ============================================================

def precompute_all_scores(dataset, model_wrapper, i3d_extractor, vggish_extractor, device):
    all_a2v, all_v2a, all_samples = [], [], []
    failed = 0

    for idx in tqdm(range(len(dataset)), desc="Pre-computing PEAVS retrieval scores"):
        sample = dataset[idx]
        try:
            s_a2v, s_v2a = get_scores_for_video(
                sample, model_wrapper, i3d_extractor, vggish_extractor, device
            )
            if s_a2v is None or s_v2a is None:
                failed += 1
                continue
            all_a2v.append(s_a2v)
            all_v2a.append(s_v2a)
            all_samples.append(sample)
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Warning: {sample['video_path']}: {e}")

    print(f"Scores computed: {len(all_samples)} ok, {failed} failed")
    return all_a2v, all_v2a, all_samples


# ============================================================
# Phase 2a: Temporal-Only
# ============================================================

def evaluate_temporal_only(all_a2v, all_v2a, all_samples, ks):
    metrics = _empty_metrics(ks)
    for i in range(len(all_samples)):
        rel = all_samples[i]["relevance"].float()
        s_a2v = all_a2v[i]
        s_v2a = all_v2a[i]

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

def evaluate_mixed_retrieval(all_a2v, all_v2a, all_samples, num_distractors, ks, seed=42):
    rng = random.Random(seed)
    n = len(all_samples)
    actual_m = min(num_distractors, n - 1)
    metrics = _empty_metrics(ks)

    for qi in tqdm(range(n), desc=f"Mixed retrieval (M={actual_m + 1} videos)"):
        rel_local = all_samples[qi]["relevance"].numpy().astype(np.float32).tolist()

        others = [i for i in range(n) if i != qi]
        distractors = rng.sample(others, actual_m)

        s_a2v = all_a2v[qi].clone().float()
        s_v2a = all_v2a[qi].clone().float()
        rel = list(rel_local)

        for di in distractors:
            nd = int(all_a2v[di].numel())
            s_a2v = torch.cat([s_a2v, torch.full((nd,), -1e9, dtype=torch.float32)], dim=0)
            s_v2a = torch.cat([s_v2a, torch.full((nd,), -1e9, dtype=torch.float32)], dim=0)
            rel.extend([0.0] * nd)

        rel_t = torch.tensor(rel, dtype=torch.float32)

        for k in ks:
            metrics["recall_a2v"][k] += recall_at_k(rel_t, s_a2v, k)
            metrics["ndcg_a2v"][k] += ndcg_at_k(rel_t, s_a2v, k)
            metrics["recall_v2a"][k] += recall_at_k(rel_t, s_v2a, k)
            metrics["ndcg_v2a"][k] += ndcg_at_k(rel_t, s_v2a, k)
        metrics["count"] += 1

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
    print(f"  {name} & {cols} \\\\")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PEAVS Temporal + Mixed Retrieval",
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
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--flow_type", type=str, default="raft", choices=["raft", "pwc"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"
    else:
        device = args.device

    model_wrapper, i3d_extractor, vggish_extractor = load_peavs(device, flow_type=args.flow_type)

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

    all_a2v, all_v2a, all_samples = precompute_all_scores(
        dataset, model_wrapper, i3d_extractor, vggish_extractor, device
    )

    if len(all_a2v) == 0:
        raise RuntimeError("No valid PEAVS scores were computed. Please check ffmpeg/model/checkpoints.")

    seg_counts = [s.numel() for s in all_a2v]
    print(f"Segments/video: min={min(seg_counts)}, max={max(seg_counts)}, mean={np.mean(seg_counts):.1f}")

    t_m = evaluate_temporal_only(all_a2v, all_v2a, all_samples, ks)
    print_metrics(t_m, ks, f"Temporal-Only ({np.mean(seg_counts):.0f} cands/query)")

    results = []
    for m in args.num_distractors:
        am = min(m, len(all_samples) - 1)
        pool = np.mean(seg_counts) * (am + 1)
        mm = evaluate_mixed_retrieval(
            all_a2v,
            all_v2a,
            all_samples,
            num_distractors=am,
            ks=ks,
            seed=args.seed,
        )
        print_metrics(mm, ks, f"Mixed: {am + 1} videos (~{pool:.0f} cands/query)")
        results.append((am + 1, pool, mm))

    print(f"\n{'=' * 65}")
    print("  LaTeX-friendly summary")
    print(f"{'=' * 65}")
    ct = max(t_m["count"], 1)
    print_latex_row(t_m, ks, "PEAVS (temporal)", ct)
    for nv, _, mm in results:
        cm = max(mm["count"], 1)
        print_latex_row(mm, ks, f"PEAVS (M={nv})", cm)


if __name__ == "__main__":
    main()
