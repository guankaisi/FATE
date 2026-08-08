from models.pe_av import PeAudioVideoModel, PeAudioVideoConfig, PeAudioVideoProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ==================== Dataset ====================

class AVE_Dataset(Dataset):
    """AVE Dataset"""
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
                if not os.path.exists(ann['video_path']):
                    continue
                if self.skip_full_match and ann['start_time'] == 0 and ann['end_time'] == 10:
                    skipped += 1
                    continue
                if ann['start_time'] == ann['end_time']:
                    skipped += 1
                    continue
                kept.append(ann)

            self.annotations = kept
            self.video_list = [ann['video_path'] for ann in self.annotations]

            if is_main_process():
                print(f"AVE Dataset: {len(all_annotations)} total, skipped {skipped}, kept {len(self.annotations)}")
        else:
            self.video_list = self._load_video_list()
            self.annotations = None

    def __len__(self):
        return len(self.video_list)

    def _load_video_list(self):
        video_files = [f for f in os.listdir(self.videos_source) if f.endswith('.mp4')]
        return [os.path.join(self.videos_source, f) for f in video_files]

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
        result = {"video_path": self.video_list[idx], "idx": idx}
        if self.annotations is not None:
            ann = self.annotations[idx]
            result['label'] = ann['label']
            result['category'] = ann['category']
            result['video_id'] = ann['video_id']
            result['start_time'] = ann['start_time']
            result['end_time'] = ann['end_time']
        return result


class AVE_Train_Dataset(AVE_Dataset):
    """AVE 训练数据集 - 排除全匹配"""
    def __init__(self, videos_source, annotation_file, skip_full_match=True):
        super().__init__(videos_source, annotation_file, skip_full_match)


# ==================== Projection Head ====================

class AVEProjectionHead(nn.Module):
    """映射头"""
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256):
        super().__init__()

        self.video_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # 可学习温度参数，初始化 log(1/0.07) ≈ 2.66
        self.logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))

    def forward(self, video_emb, audio_emb):
        v = self.video_proj(video_emb)
        a = self.audio_proj(audio_emb)
        # 不在这里归一化，让 loss 函数自己处理
        return v, a

    @property
    def temperature(self):
        # clamp 防止温度太小导致数值不稳定
        return torch.clamp(self.logit_scale.exp(), max=100.0)


# ==================== Model ====================

class PE_AV_AVE(nn.Module):
    def __init__(self, model_path=None, lora_path=None, device=None):
        super().__init__()
        self.config = PeAudioVideoConfig.from_pretrained(model_path)
        self.model = PeAudioVideoModel.from_pretrained(model_path, config=self.config)
        self.processor = PeAudioVideoProcessor.from_pretrained(model_path)

        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            if is_main_process():
                print(f"Loaded LoRA weights from {lora_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)

        # 冻结基础模型
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 可训练的映射头
        self.projection_head = AVEProjectionHead(
            input_dim=1024,
            hidden_dim=512,
            output_dim=256,
        ).to(self.device)

        if is_main_process():
            base_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            proj_trainable = sum(p.numel() for p in self.projection_head.parameters() if p.requires_grad)
            print(f"Base model trainable: {base_trainable} (should be 0)")
            print(f"Projection head trainable: {proj_trainable}")

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

        # 关键修复：用 torch.no_grad() 而不是 torch.inference_mode()
        # inference_mode 会让输出 tensor 变成 InferenceTensor，不能参与后续梯度计算
        with torch.no_grad():
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

    def extract_features(self, video_path):
        """
        从视频路径提取特征，返回普通 tensor（可参与梯度计算）
        """
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

            inputs = self.process_video_and_audio(v_clip, audio_np)

            outputs = self.encode_10s_video(
                inputs['pixel_values_videos'],
                inputs['input_values']
            )

            # .detach() 确保基础模型的梯度不回传，但 tensor 本身可以被映射头使用
            video_feat = outputs['video_embeds'].detach()
            audio_feat = outputs['audio_embeds'].detach()

            return video_feat, audio_feat

        except Exception as e:
            if is_main_process():
                print(f"Error loading {video_path}: {e}")
            return None, None

    def forward_with_projection(self, video_features, audio_features):
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(0)
            audio_features = audio_features.unsqueeze(0)

        v_proj, a_proj = self.projection_head(video_features, audio_features)
        return v_proj, a_proj


# ==================== Loss Functions ====================

def contrastive_loss(video_emb, audio_emb, labels, logit_scale):
    """
    修复后的对比损失
    """
    B, T, D = video_emb.shape

    # 归一化
    v = F.normalize(video_emb, dim=-1)
    a = F.normalize(audio_emb, dim=-1)

    # 逐秒余弦相似度
    similarity = torch.sum(v * a, dim=-1)  # [B, T], 范围 [-1, 1]

    # 用 BCE loss，target 就是 labels
    loss = F.binary_cross_entropy_with_logits(
        similarity * logit_scale,  # 缩放到合适范围
        labels,
    )

    return loss


def triplet_loss_ave(video_emb, audio_emb, labels, margin=0.3):
    """三元组损失"""
    B, T, D = video_emb.shape

    v = F.normalize(video_emb, dim=-1)
    a = F.normalize(audio_emb, dim=-1)

    total_loss = torch.tensor(0.0, device=video_emb.device)
    valid_count = 0

    for b in range(B):
        label = labels[b]
        pos_idx = torch.where(label == 1)[0]
        neg_idx = torch.where(label == 0)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        # 正样本相似度
        pos_sim = torch.sum(v[b, pos_idx] * a[b, pos_idx], dim=-1).mean()

        # 负样本相似度（随机采样一些，避免计算量太大）
        num_neg_sample = min(len(neg_idx), 3)
        neg_sample_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg_sample]]

        for p in pos_idx[:2]:  # 取前2个正样本
            neg_sim = torch.sum(v[b, p:p+1] * a[b, neg_sample_idx], dim=-1).mean()
            loss = F.relu(neg_sim - pos_sim + margin)
            total_loss = total_loss + loss
            valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=video_emb.device, requires_grad=True)

    return total_loss / valid_count


def cross_modal_nce_loss(video_emb, audio_emb, labels, logit_scale):
    """跨模态 NCE 损失"""
    B, T, D = video_emb.shape

    v = F.normalize(video_emb, dim=-1)
    a = F.normalize(audio_emb, dim=-1)

    batch_v_pos = []
    batch_a_pos = []

    for b in range(B):
        label = labels[b]
        pos_idx = torch.where(label == 1)[0]

        if len(pos_idx) > 0:
            v_pos = v[b, pos_idx].mean(dim=0)
            a_pos = a[b, pos_idx].mean(dim=0)
            batch_v_pos.append(F.normalize(v_pos, dim=0))
            batch_a_pos.append(F.normalize(a_pos, dim=0))

    if len(batch_v_pos) < 2:
        return torch.tensor(0.0, device=video_emb.device, requires_grad=True)

    batch_v_pos = torch.stack(batch_v_pos)
    batch_a_pos = torch.stack(batch_a_pos)

    sim_matrix = torch.mm(batch_v_pos, batch_a_pos.t()) * logit_scale
    labels_nce = torch.arange(len(batch_v_pos), device=video_emb.device)

    loss_v2a = F.cross_entropy(sim_matrix, labels_nce)
    loss_a2v = F.cross_entropy(sim_matrix.t(), labels_nce)

    return (loss_v2a + loss_a2v) / 2


# ==================== Training ====================

def train_projection_head(
    model,
    train_dataset,
    val_dataset=None,
    epochs=20,
    batch_size=8,
    lr=1e-4,
    save_dir="./checkpoints_ave",
    local_rank=0,
    world_size=1,
):
    """训练映射头（支持多卡）"""
    device = torch.device(f'cuda:{local_rank}')
    os.makedirs(save_dir, exist_ok=True)

    # ========== 冻结基础模型 ==========
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad = False

    # ========== 映射头设为训练模式 ==========
    model.projection_head.train()
    for param in model.projection_head.parameters():
        param.requires_grad = True

    if is_main_process():
        base_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        proj_trainable = sum(p.numel() for p in model.projection_head.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"  Base model trainable params: {base_trainable} (MUST be 0)")
        print(f"  Projection head trainable params: {proj_trainable}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        print(f"  World size: {world_size}")
        print(f"{'='*60}\n")

    # DDP 包装映射头
    if world_size > 1:
        proj_ddp = DDP(
            model.projection_head,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        proj_ddp = model.projection_head

    # 优化器：只优化映射头
    optimizer = torch.optim.AdamW(
        proj_ddp.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # DataLoader
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,  # 因为特征提取需要 GPU，不能用多 worker
        pin_memory=False,
    )

    best_val_acc = 0

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        proj_ddp.train()
        model.model.eval()  # 确保每轮都是 eval

        total_loss = 0
        total_loss_contrastive = 0
        total_loss_triplet = 0
        total_loss_nce = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",
                     disable=not is_main_process())

        for batch in pbar:
            # 提取特征（在当前进程中用 GPU 提取）
            batch_video_features = []
            batch_audio_features = []
            batch_labels = []

            for i in range(len(batch['video_path'])):
                video_path = batch['video_path'][i]
                label = batch['label'][i]  # [10]

                video_feat, audio_feat = model.extract_features(video_path)

                if video_feat is not None:
                    batch_video_features.append(video_feat)
                    batch_audio_features.append(audio_feat)
                    batch_labels.append(label)

            if len(batch_video_features) == 0:
                continue

            # Stack
            video_features = torch.stack(batch_video_features).to(device)  # [B, 10, D]
            audio_features = torch.stack(batch_audio_features).to(device)  # [B, 10, D]
            labels = torch.stack(batch_labels).to(device)                   # [B, 10]

            # Forward through projection head
            v_proj, a_proj = proj_ddp(video_features, audio_features)

            # 获取温度
            if world_size > 1:
                logit_scale = proj_ddp.module.logit_scale.exp().clamp(max=100.0)
            else:
                logit_scale = proj_ddp.logit_scale.exp().clamp(max=100.0)

            # 计算损失
            loss_c = contrastive_loss(v_proj, a_proj, labels, logit_scale)
            loss_t = triplet_loss_ave(v_proj, a_proj, labels)
            loss_n = cross_modal_nce_loss(v_proj, a_proj, labels, logit_scale)

            loss = loss_c + 0.5 * loss_t + 0.5 * loss_n

            optimizer.zero_grad()
            loss.backward()

            # 检查梯度（第一轮第一个 batch 时打印）
            if epoch == 0 and num_batches == 0 and is_main_process():
                print("\n--- Gradient Check ---")
                for name, param in model.projection_head.named_parameters():
                    if param.grad is not None:
                        print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
                    else:
                        print(f"  {name}: NO GRADIENT!")
                print("--- End Gradient Check ---\n")

            torch.nn.utils.clip_grad_norm_(proj_ddp.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_loss_contrastive += loss_c.item()
            total_loss_triplet += loss_t.item()
            total_loss_nce += loss_n.item()
            num_batches += 1

            if is_main_process():
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'c': f'{loss_c.item():.4f}',
                    't': f'{loss_t.item():.4f}',
                    'n': f'{loss_n.item():.4f}',
                })

        scheduler.step()

        if is_main_process() and num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total: {avg_loss:.4f} | "
                  f"Contrastive: {total_loss_contrastive/num_batches:.4f} | "
                  f"Triplet: {total_loss_triplet/num_batches:.4f} | "
                  f"NCE: {total_loss_nce/num_batches:.4f}")

            if val_dataset is not None and (epoch + 1) % 2 == 0:
                if world_size > 1:
                    model.projection_head.load_state_dict(proj_ddp.module.state_dict())

                val_results = evaluate_with_projection(model, val_dataset, device)
                a2v_acc = val_results['a2v_acc']
                v2a_acc = val_results['v2a_acc']

                print(f"  Validation - A2V: {a2v_acc:.1f}%, V2A: {v2a_acc:.1f}%")

                if (a2v_acc + v2a_acc) / 2 > best_val_acc:
                    best_val_acc = (a2v_acc + v2a_acc) / 2
                    save_path = os.path.join(save_dir, "best_projection_head.pt")
                    sd = proj_ddp.module.state_dict() if world_size > 1 else proj_ddp.state_dict()
                    torch.save(sd, save_path)
                    print(f"  Saved best model (avg acc: {best_val_acc:.1f}%)")

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"projection_head_epoch{epoch+1}.pt")
                sd = proj_ddp.module.state_dict() if world_size > 1 else proj_ddp.state_dict()
                torch.save(sd, save_path)

        if world_size > 1:
            dist.barrier()

    # 保存最终模型
    if is_main_process():
        final_path = os.path.join(save_dir, "projection_head_final.pt")
        sd = proj_ddp.module.state_dict() if world_size > 1 else proj_ddp.state_dict()
        torch.save(sd, final_path)
        print(f"Training complete. Saved to {final_path}")

    if world_size > 1:
        model.projection_head.load_state_dict(proj_ddp.module.state_dict())

    return model


# ==================== Evaluation ====================

def evaluate_with_projection(model, dataset, device='cuda'):
    """使用映射头评估"""
    model.projection_head.eval()

    all_video = []
    all_audio = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating",
                        disable=not is_main_process()):
            sample = dataset[idx]
            video_path = sample['video_path']
            label = sample['label']

            video_feat, audio_feat = model.extract_features(video_path)

            if video_feat is None:
                continue

            v_proj, a_proj = model.forward_with_projection(video_feat, audio_feat)

            all_video.append(v_proj.squeeze(0).cpu().numpy())
            all_audio.append(a_proj.squeeze(0).cpu().numpy())
            all_labels.append(label)

    video_features = np.stack(all_video)
    audio_features = np.stack(all_audio)
    labels = np.stack(all_labels)

    a2v, v2a = evaluate_ave_cross_modal(
        video_features, audio_features, labels,
        metric='cosine', verbose=False
    )

    model.projection_head.train()
    return {'a2v_acc': a2v, 'v2a_acc': v2a}


def compute_cosine_similarity(v_emb, a_emb):
    v_norm = v_emb / (np.linalg.norm(v_emb) + 1e-8)
    a_norm = a_emb / (np.linalg.norm(a_emb) + 1e-8)
    return np.dot(v_norm, a_norm)


def evaluate_ave_cross_modal(video_features, audio_features, labels,
                              metric='cosine', verbose=False):
    """AVE 跨模态定位评估"""
    N = len(labels)
    count_num = 0
    audio_count = 0
    video_count = 0

    for video_id in range(N):
        x_test = labels[video_id]
        if np.sum(x_test) == 10 or np.sum(x_test) == 0:
            continue

        count_num += 1
        seg = np.where(x_test == 1)[0]
        gt_start = seg[0]
        l = len(seg)

        x_video = video_features[video_id]
        x_audio = audio_features[video_id]

        # A2V
        scores = []
        for nn in range(10 - l + 1):
            s = sum(compute_cosine_similarity(x_video[nn + i], x_audio[seg[i]])
                    for i in range(l))
            scores.append(s)
        if np.argmax(scores) == gt_start:
            audio_count += 1

        # V2A
        scores = []
        for nn in range(10 - l + 1):
            s = sum(compute_cosine_similarity(x_video[seg[i]], x_audio[nn + i])
                    for i in range(l))
            scores.append(s)
        if np.argmax(scores) == gt_start:
            video_count += 1

    a2v_acc = audio_count * 100 / count_num if count_num > 0 else 0
    v2a_acc = video_count * 100 / count_num if count_num > 0 else 0

    if is_main_process():
        print(f"AVE Results | Samples: {count_num} | A2V: {a2v_acc:.1f}% | V2A: {v2a_acc:.1f}%")

    return a2v_acc, v2a_acc


def extract_all_features(model, dataset):
    """提取所有特征（无映射头）"""
    all_v, all_a, all_l, valid = [], [], [], []

    for idx in tqdm(range(len(dataset)), desc="Extracting",
                    disable=not is_main_process()):
        sample = dataset[idx]
        vf, af = model.extract_features(sample['video_path'])
        if vf is not None:
            all_v.append(vf.cpu().numpy())
            all_a.append(af.cpu().numpy())
            all_l.append(sample['label'])
            valid.append(idx)

    return np.stack(all_v), np.stack(all_a), np.stack(all_l), valid


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./weights/pe-av-small")
    parser.add_argument('--lora_path', type=str, default='./outputs/checkpoints/checkpoint-epoch-15')
    parser.add_argument('--train_annotation', type=str, default="./data/AVE_Dataset/trainSet.txt")
    parser.add_argument('--test_annotation', type=str, default="./data/AVE_Dataset/testSet.txt")
    parser.add_argument('--videos_source', type=str, default="./data/AVE_Dataset/AVE")
    parser.add_argument('--save_dir', type=str, default="./outputs/checkpoints_ave_projection")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--skip_baseline', default=True, action='store_true', help='Skip baseline evaluation')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    # 先做一次 dummy allreduce 来初始化 NCCL
    if world_size > 1:
        dummy = torch.zeros(1, device=f'cuda:{local_rank}')
        dist.all_reduce(dummy)
        del dummy
        if is_main_process():
            print("NCCL initialized successfully")

    train_dataset = AVE_Train_Dataset(
        videos_source=args.videos_source,
        annotation_file=args.train_annotation,
        skip_full_match=True,
    )

    test_dataset = AVE_Dataset(
        videos_source=args.videos_source,
        annotation_file=args.test_annotation,
        skip_full_match=True,
    )

    model = PE_AV_AVE(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=f'cuda:{local_rank}',
    )

    # Baseline：所有 rank 都做（自然同步），只有 rank 0 打印
    a2v_base, v2a_base = 0.0, 0.0
    if not args.skip_baseline:
        if is_main_process():
            print("\n>>> Baseline (no projection) <<<")
        # 所有 rank 都执行特征提取
        vf, af, lb, _ = extract_all_features(model, test_dataset)
        a2v_base, v2a_base = evaluate_ave_cross_modal(vf, af, lb)

    # 同步
    if world_size > 1:
        dist.barrier()

    # Train
    model = train_projection_head(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Final evaluation：所有 rank 都做，只有 rank 0 打印
    if is_main_process():
        print("\n>>> After training <<<")
    results = evaluate_with_projection(model, test_dataset)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"{'Method':<25} {'A2V':>10} {'V2A':>10}")
        print(f"{'-'*25} {'-'*10} {'-'*10}")
        print(f"{'Baseline':<25} {a2v_base:>9.1f}% {v2a_base:>9.1f}%")
        print(f"{'With Projection':<25} {results['a2v_acc']:>9.1f}% {results['v2a_acc']:>9.1f}%")
        print(f"{'='*60}")

    if world_size > 1:
        dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()

