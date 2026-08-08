#!/usr/bin/env python
"""
eval_fate_generation_scoring.py

Score generated videos from 5 generation models using FATE.
Outputs a JSON file compatible with the existing evaluation pipeline.

Usage:
    python eval_fate_generation_scoring.py \
        --model_path /data1/kaisi/models/pe-av-small \
        --lora_path /data1/kaisi/sync/checkpoints_vgg_round4/checkpoint-epoch-29 \
        --video_dir /data1/kaisi/sync/human-eval/baseline-videos \
        --output_json /data1/kaisi/sync/human-eval/merged_evaluations/fate_scores.json
"""

import os
import sys
import json
import warnings
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchaudio
from torchcodec.decoders import VideoDecoder
from safetensors.torch import load_file

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from models.pe_av import PeAudioVideoModel, PeAudioVideoProcessor, PeAudioVideoConfig
from peft import PeftModel


# ============================================================
# LoRA Loading
# ============================================================

def _find_adapter_weights_path(lora_path):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(lora_path, name)
        if os.path.exists(p):
            return p
    return None


def _load_adapter_state_keys(weights_path):
    if weights_path is None:
        return set()
    if weights_path.endswith(".safetensors"):
        return set(load_file(weights_path).keys())
    return set(torch.load(weights_path, map_location="cpu").keys())


def sanitize_adapter_config_for_loading(lora_path):
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    modules_to_save = config.get("modules_to_save")
    if not modules_to_save:
        return
    weights_path = _find_adapter_weights_path(lora_path)
    adapter_keys = _load_adapter_state_keys(weights_path)
    if not adapter_keys:
        return
    missing = [
        m for m in modules_to_save
        if not any(
            k.startswith(f"base_model.model.{m}.") or k.startswith(f"{m}.")
            for k in adapter_keys
        )
    ]
    if not missing:
        return
    print(f"Warning: removing missing modules_to_save: {missing}")
    backup = config_path + ".bak"
    if not os.path.exists(backup):
        with open(backup, "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    config.pop("modules_to_save", None)
    with open(config_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ============================================================
# FATE Scoring
# ============================================================

def compute_fate_score(video_path, model, processor, device):
    """
    Compute FATE synchronization score for a single video.
    
    Process:
        1. Load all video frames and audio
        2. Extract frame-level embeddings
        3. Compute frame-level similarity (Eq. 5 in paper):
           S_AV = (1/T) * sum_i (F_V^(i))^T * F_A^(i)
    
    Returns:
        score: float, FATE similarity score (higher = better sync)
    """
    # 1. Load video frames
    try:
        vr = VideoDecoder(video_path)
        fps = max(vr.metadata.average_fps, 1e-3)
        num_frames = int(vr.metadata.num_frames)
        
        if num_frames == 0:
            print(f"  Warning: 0 frames in {video_path}")
            return None
        
        indices = np.arange(0, num_frames)
        # Limit frames to avoid OOM
        max_frames = 120  # ~4s at 30fps
        if len(indices) > max_frames:
            sampled_idx = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = indices[sampled_idx]
        
        v_clip = vr.get_frames_at(indices=indices).data  # [T, C, H, W]
        if v_clip.shape[0] == 0:
            print(f"  Warning: decoded 0 frames from {video_path}")
            return None
        if v_clip.dtype != torch.uint8:
            v_clip = v_clip.to(torch.uint8)
        del vr
    except Exception as e:
        print(f"  Warning: video decode failed for {video_path}: {e}")
        return None

    # 2. Load audio
    try:
        wav, sr = torchaudio.load(video_path)
        if sr != 48000:
            wav = torchaudio.transforms.Resample(sr, 48000)(wav)
            sr = 48000
        wav = wav.mean(dim=0, keepdim=True)  # mono [1, L]
        a_clip = wav[0].numpy()  # [L]
        
        if len(a_clip) == 0:
            print(f"  Warning: empty audio in {video_path}")
            return None
    except Exception as e:
        print(f"  Warning: audio load failed for {video_path}: {e}")
        return None

    # 3. Process through model
    try:
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            inputs = processor(
                videos=v_clip,
                audio=a_clip,
                return_tensors="pt",
                padding=True,
                sampling_rate=48000,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Frame-level embeddings: [1, T, D]
            v_emb = F.normalize(outputs.video_frame_embeds.float(), dim=-1)
            a_emb = F.normalize(outputs.audio_frame_embeds.float(), dim=-1)

            # Frame-level similarity: diagonal inner product averaged over time
            # v_emb: [1, T, D], a_emb: [1, T, D]
            # Element-wise multiply then sum over D -> [1, T]
            # Then mean over T -> [1]
            score = (v_emb * a_emb).sum(dim=-1).mean(dim=-1)  # [1]

            return score.item()
    except Exception as e:
        print(f"  Warning: model inference failed for {video_path}: {e}")
        return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score generated videos using FATE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str,
                        default="./weights/pe-av-small")
    parser.add_argument("--lora_path", type=str,
                        default="./outputs/checkpoints/checkpoint-epoch-29")
    parser.add_argument("--video_dir", type=str,
                        default="./eval/human_eval/baseline-videos",
                        help="Root dir containing model subdirs (bridgedit, javis, ...)")
    parser.add_argument("--output_json", type=str,
                        default="./eval/human_eval/merged_evaluations/fate_scores.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    print(f"Loading model from {args.model_path} ...")
    model = PeAudioVideoModel.from_pretrained(args.model_path)
    processor = PeAudioVideoProcessor.from_pretrained(args.model_path)

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path} ...")
        sanitize_adapter_config_for_loading(args.lora_path)
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)

    model = model.to(device).eval()

    # ---- Discover videos ----
    # Expected structure:
    # baseline-videos/
    #   bridgedit/   -> *.mp4
    #   javis/       -> *.mp4
    #   jointdit/    -> *.mp4
    #   LTX-2/       -> *.mp4
    #   Ovi/         -> *.mp4

    model_dirs = {}
    for entry in sorted(os.listdir(args.video_dir)):
        full_path = os.path.join(args.video_dir, entry)
        if os.path.isdir(full_path):
            model_dirs[entry] = full_path

    print(f"Found {len(model_dirs)} model directories: {list(model_dirs.keys())}")

    # ---- Score all videos ----
    # Output format matches existing JSON structure:
    # {
    #   "video_filename.mp4": {
    #     "fate": {
    #       "jointdit": score,
    #       "bridgedit": score,
    #       ...
    #     }
    #   }
    # }
    #
    # But we also produce a flat structure for easier downstream processing:
    # {
    #   "by_video": { "filename.mp4": { "model": "bridgedit", "fate_score": 0.82 } },
    #   "by_model": { "bridgedit": { "filename.mp4": 0.82, ... } },
    #   "merged": { "filename.mp4": { "fate": { "bridgedit": 0.82, ... } } }
    # }

    by_model = {}      # model_name -> { filename: score }
    all_results = {}    # filename -> { model_name: score }
    
    # Also build merged format compatible with existing JSON
    merged = {}         # video_basename -> { "fate": { model_name: score } }

    total_videos = 0
    total_success = 0
    total_failed = 0

    for model_name, model_dir in model_dirs.items():
        # Collect video files
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

            score = compute_fate_score(video_path, model, processor, device)

            if score is not None:
                by_model[model_name][vf] = score
                total_success += 1

                # Build per-video dict
                if vf not in all_results:
                    all_results[vf] = {}
                all_results[vf][model_name] = score

                # Build merged format (group by base video name)
                # Video filenames might be like "prompt_001.mp4" shared across models
                # Or unique per model - handle both cases
                base_name = vf
                if base_name not in merged:
                    merged[base_name] = {}
                if "fate" not in merged[base_name]:
                    merged[base_name]["fate"] = {}
                merged[base_name]["fate"][model_name] = score
            else:
                total_failed += 1

        # Print summary for this model
        scores = list(by_model[model_name].values())
        if scores:
            print(f"  {model_name}: {len(scores)} scored, "
                  f"mean={np.mean(scores):.4f}, "
                  f"std={np.std(scores):.4f}, "
                  f"min={np.min(scores):.4f}, "
                  f"max={np.max(scores):.4f}")

    # ---- Save results ----
    output = {
        "metadata": {
            "metric": "fate",
            "model_path": args.model_path,
            "lora_path": args.lora_path,
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

    # ---- Print summary table ----
    print(f"\n{'=' * 60}")
    print(f"  FATE Score Summary (per generation model)")
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
    warnings.filterwarnings("ignore", message=r"PeAudioVideoModel does not expose.*")
    main()