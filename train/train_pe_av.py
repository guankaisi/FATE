import os
import argparse
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import bitsandbytes as bnb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import matplotlib
from torchcodec.decoders import VideoDecoder
import torchaudio
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import queue
import itertools
import time

# Suppress gradient checkpointing warnings for frozen parameters (this is expected behavior)
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True", category=UserWarning)

# 假设你的模型定义在这里
from models.pe_av import PeAudioVideoModel, PeAudioVideoConfig, PeAudioVideoProcessor
from datasets import AvsyncDatasetCrossRetrieval, VGGSOundDatasetCrossRetrieval

# ==========================================
# 1. Distributed Utils
# ==========================================
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def all_gather_with_grad(t):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    world_size = dist.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return t
    
    # Use the custom function
    all_t = GatherLayer.apply(t)
    all_t = torch.cat(all_t, dim=0)
    return all_t

# ==========================================
# 1. Loss Functions
# ==========================================
class ClipLevelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def simi_between_frame_embeds(self, embeds_1, embeds_2):
        """
        Args:
            embeds_1: [B_1, T, Dim]
            embeds_2: [B_2, T, Dim]
        Returns:
            sim_matrix: [B_1, B_2]
        """
        # 1. get shapes
        t = embeds_2.shape[1]
        # 3. sim_matrix: [B_1, B_2]
        sim_matrix = torch.einsum('btd,ctd->bc', embeds_1, embeds_2)
        # 4. scaling
        sim_matrix = sim_matrix / t
        return sim_matrix

    def forward(self, audio_embeds, video_embeds, samples):
        """
        Args:
            audio_embeds: [Batch*segments, T, Dim]
            video_embeds: [Batch*segments, T, Dim]
            samples: [Batch]
        """
        gt_indices = [sample["gt_index"] for sample in samples]
        audio_embeds = audio_embeds[gt_indices]
        video_embeds = video_embeds[gt_indices]
        # 1. Gather all embeddings from all GPUs (Global Batch)
        all_audio = all_gather_with_grad(audio_embeds) # [B * G, T, D]
        all_video = all_gather_with_grad(video_embeds) # [B * G, T, D]
        # 2. Normalize 
        audio_norm = F.normalize(audio_embeds, dim=-1) # Local Audio
        video_norm = F.normalize(video_embeds, dim=-1) # Local Video
        
        all_audio_norm = F.normalize(all_audio, dim=-1) # Global Audio
        all_video_norm = F.normalize(all_video, dim=-1) # Global Video
        
        # Release original gathered tensors early to save memory
        del all_audio, all_video
        
        # 3. Similarity Matrix
        # Local Audio vs Global Video -> [B, B * G]
        logits_a2v = self.simi_between_frame_embeds(audio_norm, all_video_norm)
        logits_a2v.div_(self.temperature)  # In-place division
        
        # Local Video vs Global Audio -> [B, B * G]
        logits_v2a = self.simi_between_frame_embeds(video_norm, all_audio_norm)
        logits_v2a.div_(self.temperature)  # In-place division
        
        # Release normalized embeddings after computing logits
        del audio_norm, video_norm, all_audio_norm, all_video_norm
        
        # 4. Labels (Correct index in global batch)
        # Rank r, local index i -> Global index: r * Batch + i
        local_batch_size = audio_embeds.size(0)
        rank = dist.get_rank()
        labels = torch.arange(local_batch_size, device=audio_embeds.device) + rank * local_batch_size
        
        # 5. Bidirectional Loss
        loss_a2v = self.cross_entropy(logits_a2v, labels)
        loss_v2a = self.cross_entropy(logits_v2a, labels)
        
        # Release logits after computing loss
        del logits_a2v, logits_v2a
        
        return (loss_a2v + loss_v2a) / 2

class TemporalLevelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def _calculate_sim_frame_emb(self, query_emb, value_emb):
        """
        Calculate similarity between query frame embeddings and candidate frame embeddings
        using diagonal averaging (similar to test_pe_av_frame_sync.py)
        
        Args:
            query_emb: [T, Dim] - query frame embeddings
            value_emb: [N, T, Dim] - candidate frame embeddings (N segments)
        Returns:
            scores: [N] - similarity scores for each candidate
        """
        # Normalize embeddings
        query_emb = F.normalize(query_emb, dim=-1)  # [T, Dim]
        value_emb = F.normalize(value_emb, dim=-1)  # [N, T, Dim]
        
        # Calculate similarity matrix: (N, T, Dim) @ (Dim, T) -> (N, T, T)
        scores_matrix = torch.matmul(value_emb, query_emb.transpose(0, 1))
        
        # Extract diagonal elements: (N, T, T) -> (N, T)
        diag_scores = scores_matrix.diagonal(dim1=-2, dim2=-1)
        
        # Release scores_matrix early
        del scores_matrix
        
        # Average over time dimension: (N, T) -> (N,)
        scores = diag_scores.mean(dim=-1)
        
        return scores
    
    def forward(self, audio_embeds, video_embeds, samples):
        """
        Args:
            audio_embeds: [Total_segments, T, Dim] - all segments from all samples in batch
            video_embeds: [Total_segments, T, Dim] - all segments from all samples in batch
            samples: List[Dict] - each dict contains 'gt_index', 'relevance', and 'segments'
        Returns:
            loss: scalar tensor
        """
        # 1. Group embeddings by sample
        # Embeddings are ordered: sample 0's all segments, sample 1's all segments, ...
        segment_idx = 0
        all_sample_audio_frame_embeds = []
        all_sample_video_frame_embeds = []
        all_sample_relevance = []
        all_sample_gt_indices = []
        
        for sample in samples:
            num_segments = len(sample['segments'])
            if segment_idx + num_segments > audio_embeds.shape[0]:
                # Safety check: skip if indices exceed bounds
                break
                
            sample_audio_embeds = audio_embeds[segment_idx:segment_idx + num_segments]  # [num_seg, T, Dim]
            sample_video_embeds = video_embeds[segment_idx:segment_idx + num_segments]  # [num_seg, T, Dim]
            sample_relevance = sample['relevance'].to(audio_embeds.device)  # [num_seg]
            
            all_sample_audio_frame_embeds.append(sample_audio_embeds)
            all_sample_video_frame_embeds.append(sample_video_embeds)
            all_sample_relevance.append(sample_relevance)
            all_sample_gt_indices.append(sample['gt_index'])
            
            segment_idx += num_segments
        
        if len(all_sample_audio_frame_embeds) == 0:
            # Return zero loss if no valid samples
            return torch.tensor(0.0, device=audio_embeds.device, requires_grad=True)
        
        # 2. Process each sample separately using only local segments (no cross-GPU)
        total_loss = 0.0
        num_valid_samples = 0
        
        for batch_idx, (audio_frame_embeds, video_frame_embeds, relevance, gt_idx) in enumerate(
            zip(all_sample_audio_frame_embeds, all_sample_video_frame_embeds, 
                all_sample_relevance, all_sample_gt_indices)
        ):
            num_segments = audio_frame_embeds.shape[0]
            if num_segments < 2:
                # Skip if only one segment (need at least one negative)
                continue
            
            # Get positive (gt_index) frame embeddings for this sample
            positive_audio_frames = audio_frame_embeds[gt_idx]  # [T, Dim]
            positive_video_frames = video_frame_embeds[gt_idx]  # [T, Dim]
            
            # Calculate similarity scores using frame-level diagonal averaging
            # Audio -> Video: positive audio frames vs all video segments
            logits_a2v = self._calculate_sim_frame_emb(
                positive_audio_frames,  # [T, Dim]
                video_frame_embeds      # [num_seg, T, Dim]
            )
            logits_a2v.div_(self.temperature)  # In-place division to save memory
            
            # Video -> Audio: positive video frames vs all audio segments
            logits_v2a = self._calculate_sim_frame_emb(
                positive_video_frames,  # [T, Dim]
                audio_frame_embeds      # [num_seg, T, Dim]
            )
            logits_v2a.div_(self.temperature)  # In-place division
            
            # Release positive frames early
            del positive_audio_frames, positive_video_frames
            
            # 3. Create weighted target distribution based on relevance
            # Only use local segments, no cross-GPU candidates
            target_a2v = torch.zeros(num_segments, device=audio_embeds.device, dtype=torch.float32)
            target_v2a = torch.zeros(num_segments, device=audio_embeds.device, dtype=torch.float32)
            
            # Relevance values: 2 -> strong positive (gt_index), 1 -> weak positive (near gt), 0 -> negative
            for local_idx in range(num_segments):
                rel_value = relevance[local_idx].item()
                # Use relevance directly as weight in the target distribution
                if rel_value > 0:
                    target_a2v[local_idx] = rel_value
                    target_v2a[local_idx] = rel_value
            
            # Normalize to create probability distribution (add small epsilon for numerical stability)
            target_a2v.add_(1e-8)  # In-place addition
            target_a2v.div_(target_a2v.sum())  # In-place division
            
            target_v2a.add_(1e-8)  # In-place addition
            target_v2a.div_(target_v2a.sum())  # In-place division
            
            # 4. Calculate weighted KL divergence loss
            log_prob_a2v = F.log_softmax(logits_a2v, dim=-1)  # [num_seg]
            log_prob_v2a = F.log_softmax(logits_v2a, dim=-1)  # [num_seg]
            
            # Release logits after computing log_softmax
            del logits_a2v, logits_v2a
            
            # Expand target to match log_prob shape for KL loss
            loss_a2v = self.kl_loss(log_prob_a2v.unsqueeze(0), target_a2v.unsqueeze(0))
            loss_v2a = self.kl_loss(log_prob_v2a.unsqueeze(0), target_v2a.unsqueeze(0))
            
            # Release intermediate tensors
            del log_prob_a2v, log_prob_v2a, target_a2v, target_v2a
            
            # Average the bidirectional loss
            sample_loss = (loss_a2v + loss_v2a) / 2.0
            total_loss += sample_loss
            num_valid_samples += 1
        
        # Average over all samples
        if num_valid_samples > 0:
            return total_loss / num_valid_samples
        else:
            # Return zero loss if no valid samples
            return torch.tensor(0.0, device=audio_embeds.device, requires_grad=True)

# ==========================================
# 2. Utility Functions
# ==========================================

def get_videos_and_audios(batch):
    """
    Collate function to handle variable length audio and video.
    Optimized for memory efficiency.
    batch_keys:
        video_path
        segments
        gt_index
        relevance
    Returns:
        all_clip_videos    
        all_clip_audios
    """
    all_clip_videos = []
    all_clip_audios = []
    for sample_idx, sample in enumerate(batch):
        video_path = sample["video_path"]
        try:
            vr = VideoDecoder(video_path)
            fps = max(vr.metadata.average_fps, 1e-3)
            num_frames = int(vr.metadata.num_frames)
        except Exception as e:
            print(f"Warning: Failed to decode video {video_path}: {e}, skipping sample")
            continue
        
        try:
            wav, sr = torchaudio.load(video_path)
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                wav = resampler(wav)
                sr = 48000
            wav = wav.mean(dim=0, keepdim=True)  # [1, L], mono
        except Exception as e:
            print(f"Warning: Failed to load audio from {video_path}: {e}, skipping sample")
            del vr
            continue
        
        segments = sample["segments"]
        for seg_idx, (s, e_sec) in enumerate(segments):
            s = float(s)
            e_sec = float(e_sec)
            start_f = max(0, int(s * fps))
            end_f = max(start_f + 1, int(e_sec * fps))
            end_f = min(end_f, num_frames)
            try:
                if start_f >= num_frames or end_f <= start_f:
                    v_clip = torch.zeros((1, 3, 336, 336), dtype=torch.uint8)
                else:
                    indices = np.arange(start_f, end_f)
                    # Limit to max 60 frames by uniform sampling
                    max_frames = 60
                    if len(indices) > max_frames:
                        # Uniformly sample max_frames indices
                        sampled_idx = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
                        indices = indices[sampled_idx]
                    v_clip = vr.get_frames_at(indices=indices).data  # [T, C, H, W]
                    if v_clip.shape[0] == 0:
                        v_clip = torch.zeros((1, 3, 336, 336), dtype=torch.uint8)
                    # Convert to uint8 if not already to save memory
                    if v_clip.dtype != torch.uint8:
                        v_clip = v_clip.to(torch.uint8)
            except Exception as e:
                print(f"Warning: Failed to decode frames [{start_f}:{end_f}] for {video_path}: {e}")
                v_clip = torch.zeros((1, 3, 336, 336), dtype=torch.uint8)         
            # 音频采样点
            start_a = max(0, int(s * sr))
            end_a = max(start_a + 1, int(e_sec * sr))
            end_a = min(end_a, wav.shape[1])
            if end_a <= start_a:
                # Use 2 seconds of silent mono audio as fallback, shape [num_samples]
                a_clip = np.zeros(int(2.0 * sr), dtype=np.float32)
            else:
                # Slice mono waveform and convert to numpy 1D array [num_samples]
                a_clip = wav[:, start_a:end_a].clone()  # [1, T]
                a_clip = a_clip[0].numpy()  # Convert to numpy 1D immediately
            
            all_clip_videos.append(v_clip)
            all_clip_audios.append(a_clip)
        
        # Release video decoder and audio early
        del vr, wav
    return all_clip_videos, all_clip_audios

def collate_identity(batch):
    all_clip_videos, all_clip_audios = get_videos_and_audios(batch)
    if len(all_clip_videos) == 0 or len(all_clip_audios) == 0:
        print(f"Warning: Empty batch after processing, returning None")
        return None
    return all_clip_videos, all_clip_audios, batch


class SkipDistributedSampler(DistributedSampler):
    """
    DistributedSampler that can skip the first N samples.
    Used for resuming training from a specific step within an epoch.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False, skip_samples=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.skip_samples = skip_samples
    
    def set_skip_samples(self, skip_samples):
        """Set number of samples to skip at the start of iteration."""
        self.skip_samples = skip_samples
    
    def __iter__(self):
        # Get the original indices from parent class
        indices = list(super().__iter__())
        # Skip the first N samples
        if self.skip_samples > 0:
            indices = indices[self.skip_samples:]
        return iter(indices)
    
    def __len__(self):
        # Return reduced length when skipping
        original_len = super().__len__()
        return max(0, original_len - self.skip_samples)

def visualize_heatmap(model, dataset, processor, folder_name, device, epoch, output_dir="figs"):
    # Ensure figs directory exists
    os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
    sample0 = dataset[0]
    all_clip_videos, all_clip_audios = get_videos_and_audios([sample0])
    positive_audio_frames = all_clip_audios[0]
    positive_video_frames = all_clip_videos[0]
    inputs = processor(
        videos=positive_video_frames,
        audio=positive_audio_frames,
        return_tensors="pt",
        padding=True,
        sampling_rate=48000,
    )
    
    
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Handle DDP/FSDP model wrapper
        if isinstance(model, DDP):
            outputs = model.module(**inputs)
        elif isinstance(model, FSDP):
            outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    audio_embeds = outputs.audio_frame_embeds  # [B, T_a, D]
    video_embeds = outputs.video_frame_embeds  # [B, T_v, D]
    
    # Normalize
    audio_embeds = F.normalize(audio_embeds, dim=-1)
    video_embeds = F.normalize(video_embeds, dim=-1)
    
    # Calculate similarity matrix [B, T_a, T_v]
    sim_matrix = torch.einsum('bid,bjd->bij', audio_embeds, video_embeds)
    
    # Plot
    sim_map = sim_matrix[0].float().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_map, cmap='viridis', origin='upper', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Frame-level Audio-Video Similarity (Epoch {epoch})')
    
    # Ticks
    steps = 20
    x_ticks = np.arange(0, sim_map.shape[1], steps)
    y_ticks = np.arange(0, sim_map.shape[0], steps)
    plt.xticks(x_ticks, x_ticks, rotation=90)
    plt.yticks(y_ticks, y_ticks)
    plt.xlabel('Video Frames')
    plt.ylabel('Audio Frames')
    
    # Diagonal
    plt.plot([0, sim_map.shape[1]-1], [0, sim_map.shape[0]-1], 'r--', alpha=0.5, label='Diagonal (Sync)')
    plt.legend()
    
    save_path = os.path.join(output_dir, folder_name, f"heatmap-epoch-{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")
    
    model.train()

def visualize_unrealted_heatmap(model, dataset, processor, folder_name, device, epoch, output_dir="figs"):
    # Ensure figs directory exists
    os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
    
    sample0 = dataset[0]
    sample1 = dataset[88]
    all_clip_videos, all_clip_audios = get_videos_and_audios([sample0])
    positive_audio_frames = all_clip_audios[0]
    positive_video_frames = all_clip_videos[0]
    all_clip_videos, all_clip_audios = get_videos_and_audios([sample1])
    negative_audio_frames = all_clip_audios[1]
    negative_video_frames = all_clip_videos[1]
    inputs = processor(
        videos=positive_video_frames,
        audio=negative_audio_frames,
        return_tensors="pt",
        padding=True,
        sampling_rate=48000,
    )  
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Handle DDP/FSDP model wrapper
        if isinstance(model, DDP):
            outputs = model.module(**inputs)
        elif isinstance(model, FSDP):
            outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    audio_embeds = outputs.audio_frame_embeds  # [B, T_a, D]
    video_embeds = outputs.video_frame_embeds  # [B, T_v, D]
    
    # Normalize
    audio_embeds = F.normalize(audio_embeds, dim=-1)
    video_embeds = F.normalize(video_embeds, dim=-1)
    
    # Calculate similarity matrix [B, T_a, T_v]
    sim_matrix = torch.einsum('bid,bjd->bij', audio_embeds, video_embeds)
    
    # Plot
    sim_map = sim_matrix[0].float().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_map, cmap='viridis', origin='upper', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Frame-level Audio-Video Similarity (Epoch {epoch})')
    
    # Ticks
    steps = 20
    x_ticks = np.arange(0, sim_map.shape[1], steps)
    y_ticks = np.arange(0, sim_map.shape[0], steps)
    plt.xticks(x_ticks, x_ticks, rotation=90)
    plt.yticks(y_ticks, y_ticks)
    plt.xlabel('Video Frames')
    plt.ylabel('Audio Frames')
    
    # Diagonal
    plt.plot([0, sim_map.shape[1]-1], [0, sim_map.shape[0]-1], 'r--', alpha=0.5, label='Diagonal (Sync)')
    plt.legend()
    
    save_path = os.path.join(output_dir, folder_name, f"heatmap-epoch-{epoch}-unrelated.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")
    
    model.train()

def visualize_offset_heatmap(model, dataset, processor, folder_name, device, epoch, offset, output_dir="figs"):
    # Ensure figs directory exists
    os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
    
    sample0 = dataset[0]
    all_clip_videos, all_clip_audios = get_videos_and_audios([sample0])
    positive_audio_frames = all_clip_audios[3]
    positive_video_frames = all_clip_videos[0]
    inputs = processor(
        videos=positive_video_frames,
        audio=positive_audio_frames,
        return_tensors="pt",
        padding=True,
        sampling_rate=48000,
    )
    
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Handle DDP/FSDP model wrapper
        if isinstance(model, DDP):
            outputs = model.module(**inputs)
        elif isinstance(model, FSDP):
            outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    audio_embeds = outputs.audio_frame_embeds  # [B, T_a, D]
    video_embeds = outputs.video_frame_embeds  # [B, T_v, D]
    
    # Normalize
    audio_embeds = F.normalize(audio_embeds, dim=-1)
    video_embeds = F.normalize(video_embeds, dim=-1)
    
    # Calculate similarity matrix [B, T_a, T_v]
    sim_matrix = torch.einsum('bid,bjd->bij', audio_embeds, video_embeds)
    
    # Plot
    sim_map = sim_matrix[0].float().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_map, cmap='viridis', origin='upper', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Frame-level Audio-Video Similarity (Epoch {epoch})')
    
    # Ticks
    steps = 20
    x_ticks = np.arange(0, sim_map.shape[1], steps)
    y_ticks = np.arange(0, sim_map.shape[0], steps)
    plt.xticks(x_ticks, x_ticks, rotation=90)
    plt.yticks(y_ticks, y_ticks)
    plt.xlabel('Video Frames')
    plt.ylabel('Audio Frames')
    
    # Diagonal
    plt.plot([0, sim_map.shape[1]-1], [0, sim_map.shape[0]-1], 'r--', alpha=0.5, label='Diagonal (Sync)')
    plt.legend()
    
    save_path = os.path.join(output_dir, folder_name, f"heatmap-epoch-{epoch}-offset-{offset}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")
    
    model.train()

def visualize_all_heatmaps(model, dataset, processor, folder_name, device, epoch, output_dir="figs"):
    visualize_heatmap(model, dataset, processor, folder_name, device, epoch, output_dir)
    visualize_unrealted_heatmap(model, dataset, processor, folder_name, device, epoch, output_dir)
    visualize_offset_heatmap(model, dataset, processor, folder_name, device, epoch, 2.0, output_dir)



# ==========================================
# 3. Training Function
# ==========================================
def train(args):
    # 0. Setup DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # 1. Initialize Model & Processor
    if local_rank == 0:
        print(f"Loading model from {args.model_path}...")
    config = PeAudioVideoConfig.from_pretrained(args.model_path)
    # Convert string dtype to torch dtype for model loading
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = torch_dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)
    
    model = PeAudioVideoModel.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype
    )
    # model = PeftModel.from_pretrained(model, "/data1/kaisi/sync/checkpoints_vgg_round2/checkpoint-step-15000", is_trainable=True)
    if args.resume_from_checkpoint:
        if local_rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        if local_rank == 0:
            model.print_trainable_parameters()       
    elif args.use_lora:
        if local_rank == 0:
            print("Using LoRA for training...")
        peft_config = LoraConfig(
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            # 针对 Transformer Block 中的所有 Attention 投影层进行微调
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
            # 投影头必须全量训练，以适应新的对齐任务
            modules_to_save=["video_head", "audio_head"]
        )
        model = get_peft_model(model, peft_config)
        # model = PeftModel.from_pretrained(model, "/data1/kaisi/sync/checkpoints_vgg_round2/checkpoint-step-15000", is_trainable=True)
        if local_rank == 0:
            model.print_trainable_parameters()
    model.gradient_checkpointing_enable()         
    model.to(device)
    
    # Check if model has frozen parameters (which would require find_unused_parameters=True)
    # This must be done after potential parameter freezing
    # Directly check if any parameters don't require grad
    has_frozen_params = any(not param.requires_grad for param in model.parameters())
    
    # Count trainable vs total parameters for logging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if local_rank == 0:
        print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
        print(f"Has frozen parameters: {has_frozen_params}")
    
    # Wrap model with FSDP or DDP
    if args.use_fsdp:
        if local_rank == 0:
            print("Using FSDP (Fully Sharded Data Parallel) for training...")
        
        # Configure mixed precision for FSDP
        mixed_precision_policy = None
        # Convert string dtype to torch dtype for FSDP
        torch_dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        fsdp_dtype = torch_dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)
        
        if fsdp_dtype in [torch.bfloat16, torch.float16]:
            mixed_precision_policy = MixedPrecision(
                param_dtype=fsdp_dtype,
                reduce_dtype=fsdp_dtype,
                buffer_dtype=fsdp_dtype,
                cast_root_forward_inputs=True,
                cast_forward_inputs=True,
            )
        
        # Configure sharding strategy
        # FULL_SHARD: Shard parameters, gradients, and optimizer states (most memory efficient)
        # SHARD_GRAD_OP: Shard gradients and optimizer states only
        # NO_SHARD: Similar to DDP (no sharding)
        sharding_strategy = ShardingStrategy.FULL_SHARD if args.fsdp_sharding_strategy == "full_shard" else \
                           ShardingStrategy.SHARD_GRAD_OP if args.fsdp_sharding_strategy == "shard_grad_op" else \
                           ShardingStrategy.NO_SHARD
        
        # Auto wrap policy: wrap transformer blocks if available, otherwise use size-based
        # For PEFT models, we might want to wrap at a different level
        auto_wrap_policy = None
        if args.fsdp_auto_wrap_policy == "size_based":
            # Wrap modules larger than 1M parameters
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)
        elif args.fsdp_auto_wrap_policy == "transformer":
            # Try to use transformer auto wrap if the model has transformer blocks
            try:
                from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
                # This is a placeholder - you may need to adjust based on your model structure
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer}
                )
            except:
                auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)
        else:
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)
        
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
            limit_all_gathers=True,  # Limit all-gather operations to reduce memory pressure
        )
    else:
        if local_rank == 0:
            print("Using DDP (Distributed Data Parallel) for training...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                    find_unused_parameters=True,  # Required for LoRA with frozen base model parameters
                    gradient_as_bucket_view=True)  # Use gradient_as_bucket_view for memory efficiency
        # Note: torch.compile is disabled due to compatibility issues with:
        # - Dynamic sequence lengths (variable video/audio lengths)
        # - LoRA/PEFT models
        # - Gradient checkpointing
        # - Complex vision models (timm/SigLIP)
        # The error "Node convert_element_type_688 was invalid" is a known torch.compile limitation
    
    processor = PeAudioVideoProcessor.from_pretrained(args.model_path)

    folder_name = f"lsemantic{args.lambda_semantic}_ltemporal{args.lambda_temporal}_temp{args.temperature}"
    # 2. Prepare Data
    if local_rank == 0:
        print(f"Loading dataset from {args.train_dir}...")
    train_dataset = AvsyncDatasetCrossRetrieval(
        videos_source = args.train_dir,
    )

    # train_dataset = VGGSOundDatasetCrossRetrieval(
    #     videos_source = args.train_dir,
    # )
    train_sampler = SkipDistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_identity,
        pin_memory=False,
        persistent_workers=False,  # Disabled: incompatible with SkipDistributedSampler
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True
    )
    # Calculate total steps per epoch (without any skipping, for scheduler)
    steps_per_epoch = len(train_dataset) // (args.batch_size * dist.get_world_size())
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )
    
    # if local_rank == 0 and args.resume_from_checkpoint is None:
        # visualize_all_heatmaps(model, train_dataset, processor, folder_name, device, epoch="pretrain", output_dir="figs")
    # Loss Functions
    semantic_loss_fn = ClipLevelContrastiveLoss(temperature=args.temperature).to(device)
    temporal_loss_fn = TemporalLevelContrastiveLoss(temperature=args.temperature).to(device)
    # Calculate total training steps (accounting for gradient accumulation)
    num_update_steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    num_training_steps = args.num_epochs * num_update_steps_per_epoch
    
    if local_rank == 0:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch_size = args.batch_size * args.accumulation_steps * world_size
        print(f"Gradient accumulation steps: {args.accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size} (batch_size={args.batch_size} x accumulation_steps={args.accumulation_steps} x num_gpus={world_size})")
        print(f"Total optimization steps: {num_training_steps}")
    
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    # 4. Training Loop
    writer = None
    if local_rank == 0 and args.use_tensorboard:
        writer = SummaryWriter(log_dir=args.output_dir)
    
    if local_rank == 0:
        logging.basicConfig(
            filename=os.path.join(args.output_dir, "training.log"),
            filemode='a',
            format='%(asctime)s - %(message)s',
            level=logging.INFO,
            force=True
        )
        logging.info(f"Training started with args: {args}")
    # if local_rank == 0:
    #     visualize_all_heatmaps(model, train_dataset, processor, folder_name, device, epoch="pretrain", output_dir="figs")
    model.train()
    global_step = 0
    
    start_epoch = 0
    skip_steps_in_epoch = 0  # Number of steps to skip at the beginning of start_epoch
    
    if args.resume_from_checkpoint:
        # Load training state (optimizer, scheduler, epoch, step) if available
        training_state_path = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        basename = os.path.basename(args.resume_from_checkpoint)
        is_epoch_checkpoint = "checkpoint-epoch-" in basename
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            lr_scheduler.load_state_dict(training_state['scheduler_state_dict'])
            global_step = training_state['global_step']
            start_epoch = training_state['epoch']

            if is_epoch_checkpoint:
                # Epoch checkpoint: resume from next epoch, no step skipping
                start_epoch = start_epoch + 1
                skip_steps_in_epoch = 0
                if local_rank == 0:
                    print(f"Loaded training state: epoch={start_epoch - 1}, global_step={global_step}, lr={optimizer.param_groups[0]['lr']:.8f}")
                    print(f"Resuming from epoch checkpoint, starting at epoch {start_epoch}")
            else:
                # Step checkpoint: resume within epoch and skip processed steps
                steps_per_epoch = len(train_dataloader)
                skip_steps_in_epoch = global_step - (start_epoch * steps_per_epoch)

                if local_rank == 0:
                    print(f"Loaded training state: epoch={start_epoch}, global_step={global_step}, lr={optimizer.param_groups[0]['lr']:.8f}")
                    print(f"Skipping first {skip_steps_in_epoch} steps in epoch {start_epoch}")
        else:
            # Fallback: parse from checkpoint name (for backward compatibility)
            if local_rank == 0:
                print(f"Warning: training_state.pt not found, parsing from checkpoint name")
            if "checkpoint-step-" in basename:
                resumed_step = int(basename.split("-")[-1])
                global_step = resumed_step
                steps_per_epoch = len(train_dataloader)
                start_epoch = resumed_step // steps_per_epoch
                skip_steps_in_epoch = resumed_step % steps_per_epoch
                if local_rank == 0:
                    print(f"Resuming from step {resumed_step}, epoch {start_epoch}, skipping first {skip_steps_in_epoch} steps in epoch")
            elif "checkpoint-epoch-" in basename:
                start_epoch = int(basename.split("-")[-1]) + 1
                if local_rank == 0:
                    print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch) # Important for shuffling
        
        # Determine how many samples to skip at the start of this epoch
        # skip_steps_in_epoch is in batches, convert to samples for sampler
        if epoch == start_epoch and skip_steps_in_epoch > 0:
            samples_to_skip = skip_steps_in_epoch * args.batch_size
            train_sampler.set_skip_samples(samples_to_skip)
            if local_rank == 0:
                print(f"Skipping first {skip_steps_in_epoch} batches ({samples_to_skip} samples) - no data loading")
        else:
            train_sampler.set_skip_samples(0)
        
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            progress_bar = tqdm(total=len(train_dataloader))
        
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            # Skip None batches (failed collation)
            if batch is None:
                print("Warning: Skipping None batch")
                if local_rank == 0:
                    progress_bar.update(1)
                global_step += 1
                continue
            
            # all_clip_videos, all_clip_audios = get_videos_and_audios(batch)
            # samples = batch
            all_clip_videos, all_clip_audios, samples = batch
            inputs = processor(
                videos=all_clip_videos,
                audio=all_clip_audios,
                return_tensors="pt",
                padding=True,
                sampling_rate=48000,
            )

            # Free CPU memory immediately after processor (before moving to GPU)
            del all_clip_videos, all_clip_audios
            
            # Move inputs to GPU
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device, non_blocking=True)
            
            # Forward with gradient accumulation support
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                
                audio_frame_embeds = outputs.audio_frame_embeds
                video_frame_embeds = outputs.video_frame_embeds
                
                # Delete inputs and outputs immediately to free memory (we only need embeddings)
                del inputs, outputs
                # Clear cache after deleting large tensors
                # torch.cuda.empty_cache()
                
                # Compute losses sequentially to reduce peak memory
                loss_temporal = None
                loss_semantic = None
                # Compute frame loss first if needed
                # Compute clip loss if needed (embeddings still available)
                if args.lambda_semantic > 0:
                    loss_semantic = semantic_loss_fn(audio_frame_embeds, video_frame_embeds, samples)

                if args.lambda_temporal > 0:
                    loss_temporal = temporal_loss_fn(audio_frame_embeds, video_frame_embeds, samples)
                
                # Now we can safely delete embeddings as both losses are computed
                del audio_frame_embeds, video_frame_embeds
                
                # Combine losses
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                if loss_temporal is not None:
                    loss = loss + args.lambda_temporal * loss_temporal
                if loss_semantic is not None:
                    loss = loss + args.lambda_semantic * loss_semantic
                
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / args.accumulation_steps
            scaled_loss.backward()
            
            # Get loss values (detach to avoid keeping computation graph)
            loss_val = loss.detach().item()
            loss_semantic_val = loss_semantic.detach().item() if loss_semantic is not None else 0.0
            loss_temporal_val = loss_temporal.detach().item() if loss_temporal is not None else 0.0
            # Delete loss tensors to free memory immediately
            del loss, scaled_loss
            if loss_semantic is not None:
                del loss_semantic
            if loss_temporal is not None:
                del loss_temporal
            # del audio_frame_embeds, video_frame_embeds
            
            total_loss += loss_val
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % args.accumulation_steps == 0:
                # Clip gradients to prevent explosion (optional but helps stability)
                if args.max_grad_norm > 0:
                    if args.use_fsdp:
                        model.clip_grad_norm_(args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            if local_rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": loss_val, 
                    "l_semantic": loss_semantic_val, 
                    "l_temporal": loss_temporal_val
                })
                
                # TensorBoard Logging (optional)
                if writer:
                    writer.add_scalar("Loss/total", loss_val, global_step)
                    writer.add_scalar("Loss/semantic", loss_semantic_val, global_step)
                    writer.add_scalar("Loss/temporal", loss_temporal_val, global_step)
                    writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
                
                # Log to file every 10 steps
                if global_step % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss_val:.6f}, Semantic: {loss_semantic_val:.6f}, Temporal: {loss_temporal_val:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
                
            global_step += 1
            
            # Note: Removed periodic torch.cuda.empty_cache() and synchronize() calls
            # These cause significant slowdown by forcing GPU synchronization
            # PyTorch's memory allocator handles memory efficiently without manual intervention
            
            # Handle remaining gradients at the end of epoch (if not divisible by accumulation_steps)
            if (batch_idx + 1) == len(train_dataloader) and (batch_idx + 1) % args.accumulation_steps != 0:
                if args.max_grad_norm > 0:
                    if args.use_fsdp:
                        model.clip_grad_norm_(args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Save checkpoint every 300 steps
            if global_step % 300 == 0 and local_rank == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save training state (optimizer, scheduler, epoch, step)
                training_state = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                }
                torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
                
                if args.use_fsdp:
                    if args.use_lora:
                        with FSDP.summon_full_params(model):
                            model.save_pretrained(checkpoint_dir)
                        print(f"Saved FSDP+LoRA checkpoint to {checkpoint_dir}")
                    else:
                        save_path = os.path.join(checkpoint_dir, "model.pt")
                        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                            state_dict = model.state_dict()
                            torch.save(state_dict, save_path)
                        print(f"Saved FSDP checkpoint to {checkpoint_dir}")
                else:
                    if args.use_lora:
                        model.module.save_pretrained(checkpoint_dir)
                        print(f"Saved LoRA checkpoint to {checkpoint_dir}")
                    else:
                        save_path = os.path.join(checkpoint_dir, "model.pt")
                        torch.save(model.module.state_dict(), save_path)
                        print(f"Saved checkpoint to {checkpoint_dir}")

        if local_rank == 0:
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

            # Save Checkpoint
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save training state (optimizer, scheduler, epoch, step)
            training_state = {
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }
            torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
            
            if args.use_fsdp:
                # FSDP requires special handling for state dict
                if args.use_lora:
                    # For FSDP + LoRA, we need to gather full state dict
                    with FSDP.summon_full_params(model):
                        model.save_pretrained(checkpoint_dir)
                    print(f"Saved FSDP+LoRA checkpoint to {checkpoint_dir}")
                else:
                    save_path = os.path.join(checkpoint_dir, "model.pt")
                    # Gather full state dict from all shards
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                        state_dict = model.state_dict()
                        torch.save(state_dict, save_path)
                    print(f"Saved FSDP checkpoint to {checkpoint_dir}")
            else:
                # DDP checkpoint saving
                if args.use_lora:
                    model.module.save_pretrained(checkpoint_dir)
                    print(f"Saved LoRA checkpoint to {checkpoint_dir}")
                else:
                    save_path = os.path.join(checkpoint_dir, "model.pt")
                    torch.save(model.module.state_dict(), save_path)
                    print(f"Saved checkpoint to {checkpoint_dir}")
        # if local_rank == 0:
        #    visualize_all_heatmaps(model, train_dataset, processor, folder_name, device, epoch, output_dir="figs")
    if local_rank == 0 and writer:
        writer.close()
        
# ==========================================
# 4. Main Entry Point
# ==========================================
def main():
    # Initialize distributed training (DDP or FSDP)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Enable TF32 for faster matmul on Ampere+ GPUs (A100, etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cudnn benchmark for optimized convolution algorithms
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="Train PE-AV Model with Frame-level Contrastive Loss")
    
    # Data arguments
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="Torch dtype")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging (disabled by default)")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="./weights/pe-av-small", help="Path to pretrained model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * accumulation_steps * num_gpus)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps for scheduler")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on") # Removed, handled by DDP
    
    # Loss arguments
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--lambda_semantic", type=float, default=0.5, help="Weight for Semantic-level loss")
    parser.add_argument("--lambda_temporal", type=float, default=1.0, help="Weight for Temporal-level loss")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    # FSDP arguments
    parser.add_argument("--use_fsdp", action="store_true", help="Use FSDP instead of DDP for training")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full_shard", 
                        choices=["full_shard", "shard_grad_op", "no_shard"],
                        help="FSDP sharding strategy: full_shard (most memory efficient), shard_grad_op, or no_shard")
    parser.add_argument("--fsdp_auto_wrap_policy", type=str, default="size_based",
                        choices=["size_based", "transformer"],
                        help="FSDP auto wrap policy: size_based or transformer")
    parser.add_argument("--fsdp_min_num_params", type=int, default=1000000,
                        help="Minimum number of parameters for size-based auto wrap policy (smaller = more granular sharding, reduces activation memory)")
    
    # Gradient clipping
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping (0 to disable)")
    
    args = parser.parse_args()
    
    # Create output directory if not exists
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Print arguments
        print("Training arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        
    train(args)

if __name__ == "__main__":
    main()
