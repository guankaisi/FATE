#!/usr/bin/env python
"""
eval_peavs_generation_scoring.py

Score generated videos from 5 generation models using PEAVS SyncMetric.
Outputs a JSON file compatible with the consistency evaluation pipeline.

Usage:
    python eval_peavs_generation_scoring.py \
        --video_dir /data1/kaisi/sync/human-eval/baseline-videos \
        --output_json /data1/kaisi/sync/human-eval/merged_evaluations/peavs_scores.json
"""

import os
import sys
import json
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
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_BASELINES = os.environ.get("FATE_BASELINE_ROOT", "/data1/kaisi/sync/baselines")
BASELINES_LIB_ROOT = ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "baselines", "avgen-eval-toolkit")) else os.path.dirname(EXTERNAL_BASELINES)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(BASELINES_LIB_ROOT, "baselines"))

PEAVS_ROOT = os.environ.get(
    "PEAVS_ROOT",
    os.path.join(BASELINES_LIB_ROOT, "baselines", "avgen-eval-toolkit", "PEAVS"),
)
PEAVS_FEAT_ROOT = os.path.join(PEAVS_ROOT, "av_features_extraction")
PEAVS_FEAT_UTILS_DIR = os.path.join(PEAVS_FEAT_ROOT, "utils")
sys.path.insert(0, PEAVS_ROOT)
sys.path.insert(0, PEAVS_FEAT_ROOT)


# ============================================================
# PEAVS Module Registration (same as retrieval script)
# ============================================================

def _register_peavs_feature_utils():
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

    with _pushd(PEAVS_FEAT_ROOT):
        i3d_extractor = ExtractI3D(i3d_args)
        vggish_extractor = ExtractVGGish(vggish_args)

    return wrapper, i3d_extractor, vggish_extractor


# ============================================================
# PEAVS Scoring
# ============================================================

def extract_features(video_path, i3d_extractor, vggish_extractor):
    """
    Extract I3D (video) and VGGish (audio) features from a video file.
    Returns:
        a_feat: [T_a, 128] numpy array
        v_feat: [T_v, 1024] numpy array (rgb + flow)
    """
    with _pushd(PEAVS_FEAT_ROOT):
        try:
            a_dict = vggish_extractor.extract(video_path)
            a_feat = np.asarray(a_dict["vggish"], dtype=np.float32)
        except Exception:
            return None, None

        try:
            v_dict = i3d_extractor.extract(video_path)
            rgb = np.asarray(v_dict["rgb"], dtype=np.float32)
            flow = np.asarray(v_dict["flow"], dtype=np.float32)
            if rgb.shape != flow.shape:
                return None, None
            v_feat = rgb + flow
        except Exception:
            return None, None

    if a_feat is None or v_feat is None:
        return None, None
    if a_feat.ndim != 2 or v_feat.ndim != 2:
        return None, None
    if a_feat.shape[0] == 0 or v_feat.shape[0] == 0:
        return None, None
    if a_feat.shape[1] != 128 or v_feat.shape[1] != 1024:
        return None, None

    return a_feat, v_feat


def compute_peavs_score(video_path, model_wrapper, i3d_extractor, vggish_extractor, device):
    """
    Compute PEAVS SyncMetric score for a single video.
    
    Returns:
        score: float (1-5 scale), or None if failed
    """
    a_feat, v_feat = extract_features(video_path, i3d_extractor, vggish_extractor)
    if a_feat is None or v_feat is None:
        return None

    try:
        xa = torch.from_numpy(a_feat).unsqueeze(0).to(device=device, dtype=torch.float32)
        xv = torch.from_numpy(v_feat).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_a = torch.ones((1, xa.shape[1]), device=device, dtype=torch.float32)
        mask_v = torch.ones((1, xv.shape[1]), device=device, dtype=torch.float32)

        pred = model_wrapper.feed_forward(xa, xv, aud_mask=mask_a, vid_mask=mask_v)
        pred = torch.clamp(pred * 4.0 + 1.0, min=1.0, max=5.0)
        return float(pred.squeeze().item())
    except Exception:
        return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score generated videos using PEAVS SyncMetric",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_dir", type=str,
                        default="./eval/human_eval/baseline-videos",
                        help="Root dir containing model subdirs (bridgedit, javis, ...)")
    parser.add_argument("--output_json", type=str,
                        default="./eval/human_eval/merged_evaluations/peavs_scores.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--flow_type", type=str, default="raft", choices=["raft", "pwc"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"
    else:
        device = args.device

    # Load PEAVS
    model_wrapper, i3d_extractor, vggish_extractor = load_peavs(device, flow_type=args.flow_type)

    # Discover model directories
    model_dirs = {}
    for entry in sorted(os.listdir(args.video_dir)):
        full_path = os.path.join(args.video_dir, entry)
        if os.path.isdir(full_path):
            model_dirs[entry] = full_path

    print(f"Found {len(model_dirs)} model directories: {list(model_dirs.keys())}")

    # Score all videos
    by_model = {}
    merged = {}
    total_videos = 0
    total_success = 0
    total_failed = 0

    for model_name, model_dir in model_dirs.items():
        video_files = sorted([
            f for f in os.listdir(model_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.webm'))
        ])

        print(f"\n{'=' * 60}")
        print(f"  Scoring {model_name}: {len(video_files)} videos")
        print(f"{'=' * 60}")

        by_model[model_name] = {}

        for vf in tqdm(video_files, desc=f"  {model_name}"):
            video_path = os.path.join(model_dir, vf)
            total_videos += 1

            score = compute_peavs_score(
                video_path, model_wrapper, i3d_extractor, vggish_extractor, device
            )

            if score is not None:
                by_model[model_name][vf] = score
                total_success += 1

                if vf not in merged:
                    merged[vf] = {}
                if "peavs" not in merged[vf]:
                    merged[vf]["peavs"] = {}
                merged[vf]["peavs"][model_name] = score
            else:
                total_failed += 1

        # Print summary
        scores = list(by_model[model_name].values())
        if scores:
            print(f"  {model_name}: {len(scores)} scored, "
                  f"mean={np.mean(scores):.4f}, "
                  f"std={np.std(scores):.4f}, "
                  f"min={np.min(scores):.4f}, "
                  f"max={np.max(scores):.4f}")

    # Save results
    output = {
        "metadata": {
            "metric": "peavs",
            "total_videos": total_videos,
            "total_success": total_success,
            "total_failed": total_failed,
        },
        "by_model": by_model,
        "merged": merged,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Results saved to {args.output_json}")
    print(f"  Total: {total_success} scored / {total_failed} failed / {total_videos} total")
    print(f"{'=' * 60}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"  PEAVS Score Summary (per generation model)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<15} {'Count':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-' * 55}")
    for model_name in sorted(by_model.keys()):
        scores = list(by_model[model_name].values())
        if scores:
            print(f"  {model_name:<15} {len(scores):>6} "
                  f"{np.mean(scores):>8.4f} {np.std(scores):>8.4f} "
                  f"{np.min(scores):>8.4f} {np.max(scores):>8.4f}")


if __name__ == "__main__":
    main()