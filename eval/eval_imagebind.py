import os
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchaudio
import av
from torchvision import transforms
from pytorchvideo import transforms as pv_transforms
from torchvision.transforms._transforms_video import NormalizeVideo

# === ImageBind Imports ===
from imagebind import data
from imagebind.data import SpatialCrop, waveform2melspec
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class AVSyncImageBindDataset(Dataset):
    def __init__(
        self, 
        videos_source='/data1/kaisi/datasets/avsync15/test', 
        meta_file='/data1/kaisi/datasets/avsync15/test.json', 
        target_frames=2, # ImageBind 默认训练使用 2 帧 (2秒片段)
        stride=64,
    ):
        self.videos_source = videos_source
        self.target_frames = target_frames
        self.stride = stride
        self.samples = [] 
        
        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)
            
        video_name_list = os.listdir(videos_source)
        video_list = [os.path.join(videos_source, v) for v in video_name_list if v.endswith('.mp4')]
        
        # 预加载数据索引
        for video_path in tqdm(video_list, desc="📂 索引视频文件"):
            self.index_single_video(video_path)

    def index_single_video(self, video_path):
        """只索引，不加载重数据，节省内存"""
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                total_frames = stream.frames
                if total_frames == 0:
                     total_frames = int(float(stream.duration * stream.time_base) * fps)
            
            if total_frames < self.target_frames:
                return

            for start_frame in range(0, total_frames, self.stride):
                end_frame = start_frame + self.target_frames
                if end_frame > total_frames:
                    break
                
                start_time = start_frame / fps
                duration = 2.0 # ImageBind expects ~2 seconds
                
                self.samples.append({
                    'video_path': video_path,
                    'start_time': start_time,
                    'duration': duration,
                    'label': self.metadata[os.path.basename(video_path)]
                })
        except Exception as e:
            print(f"Error indexing {video_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def load_video_segment(self, video_path, start_time, duration):
        frames = []
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            
            # Seek
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
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.zeros(3, 3, 2, 224, 224)

        if not frames:
             return torch.zeros(3, 3, 2, 224, 224)

        # [T, H, W, C] -> [C, T, H, W]
        frames = torch.as_tensor(np.stack(frames)).permute(3, 0, 1, 2)
        frames = frames.float() / 255.0
        
        transform = transforms.Compose([
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        frames = transform(frames)
        
        # Subsample to 2 frames
        temporal_subsample = pv_transforms.UniformTemporalSubsample(num_samples=2)
        frames = temporal_subsample(frames)
        
        # Spatial Crop (returns list of crops)
        spatial_crop = SpatialCrop(224, num_crops=3)
        crops = spatial_crop([frames]) 
        
        return torch.stack(crops, dim=0) # [3, C, T, H, W]

    def load_audio_segment(self, video_path, start_time, duration):
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

            melspec = waveform2melspec(waveform_clip, sr, 128, 204) # [1, 128, 204]
            
            mean = -4.268
            std = 9.138
            normalize = transforms.Normalize(mean=mean, std=std)
            melspec = normalize(melspec)
            
            return melspec.unsqueeze(0).repeat(3, 1, 1, 1) # [3, 1, 128, 204]
        except Exception as e:
            print(f"Error loading audio {video_path}: {e}")
            return torch.zeros(3, 1, 128, 204)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video_path = item['video_path']
        
        video_tensor = self.load_video_segment(video_path, item['start_time'], item['duration'])
        audio_tensor = self.load_audio_segment(video_path, item['start_time'], item['duration'])
        
        return {
            "video": video_tensor, 
            "audio": audio_tensor, 
            "label": item['label']
        }

def extract_features_and_labels(model, dataloader, device):
    video_feats = []
    audio_feats = []
    text_feats = []
    all_labels = []
    
    model.eval()
    print("🚀 开始提取 ImageBind 特征 (Video, Audio, Text)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            try:
                # ImageBind 期望输入是字典格式
                # 1. Prepare Inputs
                video_input = batch['video'].to(device)
                audio_input = batch['audio'].to(device)
                labels = batch['label'] # Tuple of strings
                print(video_input.shape, audio_input.shape)
                # Text needs to be tokenized on the fly. 
                # ImageBind's load_and_transform_text handles tokenization.
                text_input = data.load_and_transform_text(labels, device)

                inputs = {
                    ModalityType.VISION: video_input,
                    ModalityType.AUDIO: audio_input,
                    ModalityType.TEXT: text_input
                }
                
                embeddings = model(inputs)
                
                v_emb = embeddings[ModalityType.VISION]
                a_emb = embeddings[ModalityType.AUDIO]
                t_emb = embeddings[ModalityType.TEXT]
                
                # Normalize embeddings (Improved based on snippet)
                v_emb = F.normalize(v_emb, dim=-1)
                a_emb = F.normalize(a_emb, dim=-1)
                t_emb = F.normalize(t_emb, dim=-1)
                
                video_feats.append(v_emb.cpu())
                audio_feats.append(a_emb.cpu())
                text_feats.append(t_emb.cpu())
                all_labels.extend(labels)
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
            
    return (
        torch.cat(video_feats), 
        torch.cat(audio_feats), 
        torch.cat(text_feats),
        all_labels
    )

# 复用你之前的测评函数
def compute_instance_retrieval(video_feats, audio_feats):
    """计算基于实例的检索准确率"""
    device = video_feats.device
    
    # Features are already normalized in extract_features_and_labels, but doing it again is safe
    video_feats = F.normalize(video_feats, p=2, dim=1)
    audio_feats = F.normalize(audio_feats, p=2, dim=1)
    
    # --- 1. Alignment Scores (Diagonal Mean) ---
    # Calculate cosine similarity for corresponding pairs
    va_sim = (video_feats * audio_feats).sum(dim=1).mean().item()
    
    print("\n📈 Alignment Scores (Avg Cosine Sim):")
    print(f"Video-Audio: {va_sim:.4f}")

    # --- 2. Retrieval Metrics (Video <-> Audio) ---
    sim_matrix = torch.matmul(video_feats, audio_feats.t())
    n_samples = sim_matrix.shape[0]
    ground_truth = torch.arange(n_samples).to(device)

    print("\n📊 计算 Video -> Audio (ImageBind Instance)...")
    topk_k = 10
    _, indices = torch.topk(sim_matrix, k=topk_k, dim=1)
    matches = (indices == ground_truth.unsqueeze(1))
    
    acc_1 = matches[:, :1].any(dim=1).float().mean().item()
    acc_5 = matches[:, :5].any(dim=1).float().mean().item()
    acc_10 = matches[:, :10].any(dim=1).float().mean().item()
    print(f"Video->Audio: R@1: {acc_1:.2%}, R@5: {acc_5:.2%}, R@10: {acc_10:.2%}")

    print("📊 计算 Audio -> Video (ImageBind Instance)...")
    sim_matrix_t = sim_matrix.t()
    _, indices = torch.topk(sim_matrix_t, k=topk_k, dim=1)
    matches = (indices == ground_truth.unsqueeze(1))
    
    acc_1_av = matches[:, :1].any(dim=1).float().mean().item()
    acc_5_av = matches[:, :5].any(dim=1).float().mean().item()
    acc_10_av = matches[:, :10].any(dim=1).float().mean().item()
    print(f"Audio->Video: R@1: {acc_1_av:.2%}, R@5: {acc_5_av:.2%}, R@10: {acc_10_av:.2%}")

    return (acc_1 + acc_1_av) / 2

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 准备模型
    print("🏗️ Loading ImageBind Model...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    
    # 2. 准备数据
    # 注意：ImageBind 的数据处理逻辑稍微不同，需要用它的 Dataset
    dataset = AVSyncImageBindDataset() 
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"总样本数 (Chunks): {len(dataset)}")
    
    # 3. 提取特征
    v_feats, a_feats, t_feats, labels = extract_features_and_labels(model, dataloader, device)
    
    # 4. 测评
    compute_instance_retrieval(v_feats, a_feats)