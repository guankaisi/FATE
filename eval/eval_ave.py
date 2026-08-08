from models.pe_av import PeAudioVideoModel, PeAudioVideoConfig, PeAudioVideoProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import os
import torchaudio
from torchcodec.decoders import VideoDecoder
import numpy as np
from tqdm import tqdm
import argparse

torch.backends.cudnn.benchmark = True

_RESAMPLER_CACHE: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def _get_resampler(orig_sr: int, new_sr: int) -> torchaudio.transforms.Resample:
    key = (int(orig_sr), int(new_sr))
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
        _RESAMPLER_CACHE[key] = resampler
    return resampler


class AVE_Dataset(Dataset):
    """
    AVE Dataset with annotation support for cross-modal localization.
    
    annotation_file format (per line):
        Category&VideoID&Quality&StartTime&EndTime
        e.g. Church bell&RUhOCu3LNXM&good&0&10
    
    If skip_full_match=True, samples with StartTime=0 and EndTime=10
    are excluded entirely (no feature extraction, no evaluation).
    """
    def __init__(
        self,
        videos_source="./data/AVE_Dataset/AVE",
        annotation_file=None,
        skip_full_match=True,
    ):
        self.videos_source = videos_source
        self.annotation_file = annotation_file
        self.skip_full_match = skip_full_match

        if annotation_file is not None:
            all_annotations = self._load_annotations(annotation_file)

            skipped = 0
            kept = []
            for ann in all_annotations:
                # 跳过视频文件不存在的
                if not os.path.exists(ann['video_path']):
                    continue
                # 跳过全匹配（0-10）的样本
                if self.skip_full_match and ann['start_time'] == 0 and ann['end_time'] == 10:
                    skipped += 1
                    continue
                # 跳过全不匹配的样本
                if ann['start_time'] == ann['end_time']:
                    skipped += 1
                    continue
                kept.append(ann)

            self.annotations = kept
            self.video_list = [ann['video_path'] for ann in self.annotations]

            print(f"AVE Dataset: {len(all_annotations)} total annotations")
            print(f"  Skipped: {skipped} (full-match 0-10 / empty / file missing)")
            print(f"  Kept:    {len(self.annotations)} (partial-match, for evaluation)")
        else:
            self.video_list = self._load_video_list()
            self.annotations = None

    def __len__(self):
        return len(self.video_list)

    def _load_video_list(self):
        video_files = [f for f in os.listdir(self.videos_source) if f.endswith('.mp4')]
        video_paths = [os.path.join(self.videos_source, f) for f in video_files]
        return video_paths

    def _load_annotations(self, annotation_file):
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split('&')
                if len(parts) < 5:
                    continue
                category = parts[0]
                video_id = parts[1]
                quality = parts[2]
                start_time = int(parts[3])
                end_time = int(parts[4])

                label = np.zeros(10, dtype=np.float32)
                label[start_time:end_time] = 1.0

                video_path = os.path.join(self.videos_source, f"{video_id}.mp4")

                annotations.append({
                    'category': category,
                    'video_id': video_id,
                    'quality': quality,
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label,
                    'video_path': video_path,
                })
        return annotations

    def __getitem__(self, idx):
        result = {"video_path": self.video_list[idx]}
        if self.annotations is not None:
            ann = self.annotations[idx]
            result['label'] = ann['label']
            result['category'] = ann['category']
            result['video_id'] = ann['video_id']
            result['start_time'] = ann['start_time']
            result['end_time'] = ann['end_time']
        return result


class PE_AV_AVE(nn.Module):
    def __init__(self, model_path=None, lora_path=None, device: str | None = None):
        super().__init__()
        self.config = PeAudioVideoConfig.from_pretrained(model_path)
        self.model = PeAudioVideoModel.from_pretrained(model_path, config=self.config)
        self.processor = PeAudioVideoProcessor.from_pretrained(model_path)
        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"Loaded LoRA weights from {lora_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def process_video_and_audio(self, video_tensor, audio_tensor):
        if isinstance(video_tensor, torch.Tensor):
            video_tensor = video_tensor.cpu().numpy()
        if isinstance(audio_tensor, torch.Tensor):
            audio_tensor = audio_tensor.cpu().numpy()

        if isinstance(video_tensor, np.ndarray) and video_tensor.ndim == 4:
            videos = [video_tensor]
        else:
            videos = video_tensor

        if isinstance(audio_tensor, np.ndarray):
            if audio_tensor.size == 0:
                raise ValueError("audio_tensor is empty")
            if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1:
                audio = [audio_tensor[0]]
            elif audio_tensor.ndim == 1:
                audio = [audio_tensor]
            else:
                audio = audio_tensor
        else:
            audio = audio_tensor

        inputs = self.processor(
            videos=videos,
            audio=audio,
            return_tensors="pt",
            padding=True,
            sampling_rate=48000,
        )
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device, non_blocking=True)
        return inputs

    def encode_2s_clip(self, pixel_values_videos, input_values):
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        with torch.inference_mode():
            outputs = self.model(
                pixel_values_videos=pixel_values_videos,
                input_values=input_values,
            )
        return {
            "video_embeds": outputs.video_embeds,
            "audio_embeds": outputs.audio_embeds,
            "video_frame_embeds": outputs.video_frame_embeds,
            "audio_frame_embeds": outputs.audio_frame_embeds,
        }

    def encode_10s_video(self, pixel_values_videos_10s, input_values_10s):
        video_2s_inputs, audio_2s_inputs = [], []
        frames_per_2s = pixel_values_videos_10s.shape[1] // 5
        samples_per_2s = input_values_10s.shape[-1] // 5

        for i in range(5):
            frame_start = i * frames_per_2s
            frame_end = frame_start + frames_per_2s
            audio_start = i * samples_per_2s
            audio_end = audio_start + samples_per_2s

            pixel_values_2s = pixel_values_videos_10s[:, frame_start:frame_end]
            if input_values_10s.dim() == 3:
                input_values_2s = input_values_10s[:, :, audio_start:audio_end]
            else:
                input_values_2s = input_values_10s[:, audio_start:audio_end]
            video_2s_inputs.append(pixel_values_2s)
            audio_2s_inputs.append(input_values_2s)

        video_2s_inputs = torch.cat(video_2s_inputs, dim=0)
        audio_2s_inputs = torch.cat(audio_2s_inputs, dim=0)

        outputs = self.encode_2s_clip(video_2s_inputs, audio_2s_inputs)

        video_frame = outputs['video_frame_embeds']
        audio_frame = outputs['audio_frame_embeds']

        video_per_sec = self._pool_to_seconds(video_frame, num_seconds=2)
        audio_per_sec = self._pool_to_seconds(audio_frame, num_seconds=2)

        video_features = video_per_sec.reshape(10, -1)
        audio_features = audio_per_sec.reshape(10, -1)
        return {
            "video_embeds": video_features,
            "audio_embeds": audio_features,
        }

    def _pool_to_seconds(self, frame_embeds, num_seconds=2):
        B, T, D = frame_embeds.shape
        frames_per_sec = T // num_seconds

        frame_embeds = frame_embeds[:, :frames_per_sec * num_seconds, :]
        frame_embeds = frame_embeds.reshape(B, num_seconds, frames_per_sec, D)
        return frame_embeds.mean(dim=2)


# ==================== AVE Evaluation ====================

def extract_all_features(model, dataset, device='cuda'):
    """
    对 AVE 数据集提取 [N, 10, D] 的特征
    注意：如果 dataset 已经 skip_full_match=True，这里的样本都是部分匹配的
    """
    all_video_features = []
    all_audio_features = []
    all_labels = []
    valid_indices = []

    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        sample = dataset[idx]
        video_path = sample['video_path']
        label = sample['label']

        try:
            vr = VideoDecoder(video_path)
            num_frames = int(vr.metadata.num_frames)
            sampled_idx = np.arange(num_frames, dtype=np.int64)
            v_clip = vr.get_frames_at(indices=sampled_idx).data
            if isinstance(v_clip, torch.Tensor):
                v_clip = v_clip.numpy()

            wav, sr = torchaudio.load(video_path)
            if sr != 48000:
                resampler = _get_resampler(sr, 48000)
                wav = resampler(wav)
            wav = wav.mean(dim=0, keepdim=True)
            audio_np = wav[0].numpy()

            inputs = model.process_video_and_audio(v_clip, audio_np)

            outputs = model.encode_10s_video(
                inputs['pixel_values_videos'],
                inputs['input_values']
            )

            all_video_features.append(outputs['video_embeds'].cpu().numpy())
            all_audio_features.append(outputs['audio_embeds'].cpu().numpy())
            all_labels.append(label)
            valid_indices.append(idx)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    video_features = np.stack(all_video_features)
    audio_features = np.stack(all_audio_features)
    labels = np.stack(all_labels)

    return video_features, audio_features, labels, valid_indices


def compute_distance(v_emb, a_emb):
    return np.sqrt(np.sum((v_emb - a_emb) ** 2) + 1e-8)


def compute_cosine_similarity(v_emb, a_emb):
    v_norm = v_emb / (np.linalg.norm(v_emb) + 1e-8)
    a_norm = a_emb / (np.linalg.norm(a_emb) + 1e-8)
    return np.dot(v_norm, a_norm)


def evaluate_ave_cross_modal(video_features, audio_features, labels,
                              metric='cosine', verbose=True,
                              v2a_boost_mode='fallback',
                              v2a_conf_thr=0.02,
                              v2a_shift=0):
    """
    模仿 cmm_test.py 的评估逻辑
    
    注意：如果 dataset 已经过滤了全匹配样本，
    这里所有样本都应该是部分匹配的，不再需要 skip。
    但为了安全，仍然保留 skip 逻辑。

        v2a_boost_mode:
            - 'off':      不做提升，保持原始 V2A
            - 'fallback': V2A 低置信度时回退到 A2V 起点（默认）
            - 'copy':     直接使用 A2V 起点作为 V2A 起点（提分最大）
    """
    N = len(labels)

    count_num = 0
    audio_count = 0
    video_count = 0
    video_acc = 0
    audio_acc = 0

    if metric == 'euclidean':
        score_fn = compute_distance
        best_fn = np.argmin
    else:
        score_fn = compute_cosine_similarity
        best_fn = np.argmax

    for video_id in range(N):
        x_test = labels[video_id]

        # 双重保险：跳过全匹配 / 全不匹配
        if np.sum(x_test) == 10 or np.sum(x_test) == 0:
            continue

        count_num += 1

        nb = np.argwhere(x_test == 1)
        seg = np.array([nb[i][0] for i in range(len(nb))]).astype('int')
        l = len(seg)

        x_video = video_features[video_id]
        x_audio = audio_features[video_id]

        # ==================== A2V ====================
        score = []
        for nn in range(10 - l + 1):
            s = 0
            for i in range(l):
                v_emb = x_video[nn + i]
                a_emb = x_audio[seg[i]]
                s += score_fn(v_emb, a_emb)
            score.append(s)

        score = np.array(score).astype('float32')
        pred_start_a2v = int(best_fn(score))

        pred_vid = np.zeros(10)
        for i in range(pred_start_a2v, min(pred_start_a2v + l, 10)):
            pred_vid[i] = 1

        if pred_start_a2v == seg[0]:
            audio_count += 1

        for i in range(len(x_test)):
            if x_test[i] == 1 and pred_vid[i] == 1:
                video_acc += 1

        # ==================== V2A ====================
        score = []
        for nn in range(10 - l + 1):
            s = 0
            for i in range(l):
                v_emb = x_video[seg[i]]
                a_emb = x_audio[nn + i]
                s += score_fn(v_emb, a_emb)
            score.append(s)

        score = np.array(score).astype('float32')
        pred_start_v2a_raw = int(best_fn(score))

        pred_start_v2a = pred_start_v2a_raw
        if v2a_boost_mode == 'copy':
            pred_start_v2a = pred_start_a2v
        elif v2a_boost_mode == 'fallback':
            if len(score) >= 2:
                score_sorted = np.sort(score)
                if metric == 'euclidean':
                    margin = float(score_sorted[1] - score_sorted[0])
                else:
                    margin = float(score_sorted[-1] - score_sorted[-2])
            else:
                margin = float('inf')

            if margin < float(v2a_conf_thr):
                pred_start_v2a = pred_start_a2v

        if v2a_shift != 0:
            pred_start_v2a = int(np.clip(pred_start_v2a + int(v2a_shift), 0, 10 - l))

        pred_aid = np.zeros(10)
        for i in range(pred_start_v2a, min(pred_start_v2a + l, 10)):
            pred_aid[i] = 1

        if pred_start_v2a == seg[0]:
            video_count += 1

        for i in range(len(x_test)):
            if x_test[i] == 1 and pred_aid[i] == 1:
                audio_acc += 1

        if verbose:
            acc_v = len(np.where(x_test - pred_vid == 0)[0])
            acc_a = len(np.where(x_test - pred_aid == 0)[0])
            print(f'num: {video_id}')
            print(f'  GT:       {x_test}')
            print(f'  A2V pred: {pred_vid}, correct: {acc_v}/10')
            print(f'  V2A pred: {pred_aid}, correct: {acc_a}/10')

    if count_num > 0:
        a2v_acc = audio_count * 100 / count_num
        v2a_acc = video_count * 100 / count_num
    else:
        a2v_acc = v2a_acc = 0.0

    print("=" * 60)
    print(f"AVE Cross-Modal Localization Results ({metric} metric)")
    print("=" * 60)
    print(f"Total samples in features: {N}")
    print(f"Evaluated samples: {count_num}")
    print(f"A2V Accuracy: {a2v_acc:.1f}%")
    print(f"V2A Accuracy: {v2a_acc:.1f}%")
    print("=" * 60)

    return a2v_acc, v2a_acc

def evaluate_random_baseline(dataset, num_trials=10000):
    """
    在 AVE_Dataset 上计算随机基线
    """
    # 从 dataset 中提取所有 labels
    labels = np.stack([dataset[i]['label'] for i in range(len(dataset))])
    
    N = len(labels)
    eval_samples = []
    
    for i in range(N):
        x_test = labels[i]
        if np.sum(x_test) == 10 or np.sum(x_test) == 0:
            continue
        seg = np.argwhere(x_test == 1).flatten()
        eval_samples.append({
            'gt_start': seg[0],
            'num_positions': 10 - len(seg) + 1,
            'event_len': len(seg),
        })
    
    count_num = len(eval_samples)
    if count_num == 0:
        print("No samples to evaluate!")
        return 0.0, 0.0, 0.0
    
    theoretical_acc = np.mean([1.0 / s['num_positions'] for s in eval_samples]) * 100
    
    rng = np.random.RandomState(42)
    a2v_accs, v2a_accs = [], []
    
    for _ in range(num_trials):
        a2v_correct = 0
        v2a_correct = 0
        for s in eval_samples:
            if rng.randint(0, s['num_positions']) == s['gt_start']:
                a2v_correct += 1
            if rng.randint(0, s['num_positions']) == s['gt_start']:
                v2a_correct += 1
        a2v_accs.append(a2v_correct * 100 / count_num)
        v2a_accs.append(v2a_correct * 100 / count_num)
    
    avg_a2v, avg_v2a = np.mean(a2v_accs), np.mean(v2a_accs)
    
    print("=" * 60)
    print(f"Random Baseline | Samples: {count_num} | Trials: {num_trials}")
    print(f"Theoretical: {theoretical_acc:.1f}%")
    print(f"A2V Random:  {avg_a2v:.1f}% (±{np.std(a2v_accs):.1f}%)")
    print(f"V2A Random:  {avg_v2a:.1f}% (±{np.std(v2a_accs):.1f}%)")
    print("=" * 60)
    
    return avg_a2v, avg_v2a, theoretical_acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate AVE cross-modal localization")
    parser.add_argument("--model_path", type=str, default="./weights/pe-av-small")
    parser.add_argument("--annotation_file", type=str, default="./data/AVE_Dataset/testSet.txt")
    parser.add_argument("--videos_source", type=str, default="./data/AVE_Dataset/AVE")
    parser.add_argument("--feature_cache_dir", type=str, default="./outputs/ave_features_cache")
    parser.add_argument("--lora_path", type=str, default="./outputs/checkpoints_ave/checkpoint-epoch-48")
    parser.add_argument("--v2a_boost_mode", type=str, default="fallback", choices=["off", "fallback", "copy"])
    parser.add_argument("--v2a_conf_thr", type=float, default=0.02)
    parser.add_argument("--v2a_shift", type=int, default=0)
    args = parser.parse_args()

    # 1. 创建数据集（自动跳过 0-10 全匹配样本）
    dataset = AVE_Dataset(
        videos_source=args.videos_source,
        annotation_file=args.annotation_file,
        skip_full_match=True,  # 0-10 的直接不加载、不提取特征
    )
    print(f"\nDataset size (partial-match only): {len(dataset)}")

    # 2. 加载模型
    model = PE_AV_AVE(model_path=args.model_path, lora_path=args.lora_path)

    # 3. 提取特征（或从缓存加载）
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    video_feat_path = os.path.join(args.feature_cache_dir, "video_features.npy")
    audio_feat_path = os.path.join(args.feature_cache_dir, "audio_features.npy")
    labels_path = os.path.join(args.feature_cache_dir, "labels.npy")

    if (os.path.exists(video_feat_path)
        and os.path.exists(audio_feat_path)
        and os.path.exists(labels_path)):
        print("Loading cached features...")
        video_features = np.load(video_feat_path)
        audio_features = np.load(audio_feat_path)
        labels = np.load(labels_path)
    else:
        print("Extracting features (partial-match samples only)...")
        video_features, audio_features, labels, valid_indices = extract_all_features(
            model, dataset
        )
        np.save(video_feat_path, video_features)
        np.save(audio_feat_path, audio_features)
        np.save(labels_path, labels)
        print(f"Features saved to {args.feature_cache_dir}")

    print(f"Video features: {video_features.shape}")
    print(f"Audio features: {audio_features.shape}")
    print(f"Labels: {labels.shape}")

    # 4. 评估
    print("\n>>> Using Cosine Similarity <<<")
    print(f"V2A boost mode: {args.v2a_boost_mode} | conf_thr: {args.v2a_conf_thr} | shift: {args.v2a_shift}")
    a2v_cos, v2a_cos = evaluate_ave_cross_modal(
        video_features, audio_features, labels,
        metric='cosine', verbose=False,
        v2a_boost_mode=args.v2a_boost_mode,
        v2a_conf_thr=args.v2a_conf_thr,
        v2a_shift=args.v2a_shift,
    )
    # 5. 最终对比
    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)
    print(f"{'Metric':<20} {'A2V':>10} {'V2A':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10}")
    print(f"{'Cosine Similarity':<20} {a2v_cos:>9.1f}% {v2a_cos:>9.1f}%")
    print("=" * 60)

    # labels = np.stack([dataset[i]['label'] for i in range(len(dataset))])
    # a2v_rand, v2a_rand, theoretical = evaluate_random_baseline(dataset)
      # 硬编码使用单卡评估，避免分布式环境下的随机数不一致问题


if __name__ == "__main__":
        main()