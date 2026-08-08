import os
import torch
import torch.nn.functional as F
import numpy as np
# from models.av_jepa import AV_JEPA
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
# from models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor
import torchaudio
import torchaudio.transforms as transforms
from torchaudio.compliance import kaldi
import torch.distributed as dist


class AvsyncDataset(Dataset):
    def __init__(
        self,
        videos_source='/data1/kaisi/datasets/train_avsync_vggss',
    ):
        self.videos_source = videos_source
        self.video_list = self._collect_mp4(videos_source)
    def __len__(self):
        return len(self.video_list)
        
    def __getitem__(self, idx):  
        return {
            "video_path": self.video_list[idx],
        }

    def _collect_mp4(self, root_dir):
        """Recursively collect mp4 files under root_dir."""
        video_paths = []
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".mp4"):
                    video_paths.append(os.path.join(r, f))
        video_paths.sort()
        if not video_paths:
            print(f"Warning: No mp4 found under {root_dir}")
        return video_paths

class AvsyncDatasetCrossRetrieval(Dataset):
    def __init__(
        self,
        videos_source="/data1/kaisi/datasets/test_avsync",
        window_size=2.0,
        stride=0.5,
        max_segments=20,  # 新增：限制每个样本的最大 segment 数量
    ):
        self.videos_source = videos_source
        self.window_size = float(window_size)
        self.stride = float(stride)
        self.max_segments = int(max_segments)  # 最多保留 10 个 segments

        self.video_list = self._collect_mp4(videos_source)
        self._duration_cache = {}

    def __len__(self):
        return len(self.video_list)

    def _collect_mp4(self, root_dir):
        """Recursively collect mp4 files under root_dir."""
        video_paths = []
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".mp4"):
                    video_paths.append(os.path.join(r, f))
        video_paths.sort()
        if not video_paths:
            print(f"Warning: No mp4 found under {root_dir}")
        return video_paths

    def _get_video_duration(self, video_path):
        """通过元数据估计时长，失败时回退 10s。"""
        if video_path in self._duration_cache:
            return self._duration_cache[video_path]
        try:
            # 使用 torchvision.io.read_video 读取视频元数据
            # from torchvision.io import read_video
            # video, _, info = read_video(video_path, pts_unit="sec")
            # if video.numel() > 0:
            #     fps = float(info.get("video_fps", 25.0) or 25.0)
            #     if fps > 0:
            #         num_frames = video.shape[0]
            #         duration = max(num_frames / fps, 0.1)
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(video_path)
            duration = decoder.metadata.duration_seconds
        except Exception:
            duration = 10.0
        self._duration_cache[video_path] = float(duration)
        return self._duration_cache[video_path]

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        duration = self._get_video_duration(video_path)

        # 动态生成切片起点
        if duration <= self.window_size:
            starts = [0.0]
        else:
            starts = []
            cur = 0.0
            limit = duration - self.window_size
            while cur <= limit + 1e-6:
                starts.append(round(cur, 4))
                cur += self.stride
            if not starts:
                starts = [0.0]

        segments = [(s, min(s + self.window_size, duration)) for s in starts]

        # 随机选 GT 段（使用按样本固定的随机数生成器，避免污染全局随机状态）
        rng = np.random.default_rng(42 + idx)
        gt_idx = int(rng.integers(len(segments)))
        gt_start = segments[gt_idx][0]

        relevance = []
        for i, (s, _) in enumerate(segments):
            if i == gt_idx:
                relevance.append(2)
            elif abs(s - gt_start) <= 0.5:
                relevance.append(1)
            else:
                relevance.append(0)

        # 筛选 segments：保留 relevance=1 和 relevance=2 的，然后从 relevance=0 中采样到总数 max_segments
        relevance_array = np.array(relevance)
        high_relevance_indices = np.where(relevance_array >= 1)[0]  # relevance=1 和 2 的索引
        low_relevance_indices = np.where(relevance_array == 0)[0]  # relevance=0 的索引
        
        # 计算需要从 relevance=0 中采样多少个
        num_high_relevance = len(high_relevance_indices)
        num_to_sample = max(0, self.max_segments - num_high_relevance)
        
        # 如果 relevance=0 的数量足够，随机采样；否则全部保留
        if len(low_relevance_indices) > 0 and num_to_sample > 0:
            if len(low_relevance_indices) <= num_to_sample:
                selected_low_indices = low_relevance_indices
            else:
                # 使用同一个按样本固定的 RNG 保证可复现
                selected_low_indices = rng.choice(
                    low_relevance_indices, 
                    size=num_to_sample, 
                    replace=False
                )
            selected_indices = np.concatenate([high_relevance_indices, selected_low_indices])
        else:
            # 如果没有 relevance=0 的，或者不需要采样，只保留高 relevance 的
            selected_indices = high_relevance_indices
        
        # 排序保持顺序
        selected_indices = np.sort(selected_indices)
        
        # 重新构建 segments 和 relevance
        filtered_segments = [segments[i] for i in selected_indices]
        filtered_relevance = [relevance[i] for i in selected_indices]
        
        # 更新 gt_index（因为 segments 列表变了）
        new_gt_idx = np.where(selected_indices == gt_idx)[0]
        if len(new_gt_idx) > 0:
            new_gt_idx = int(new_gt_idx[0])
        else:
            # 如果 GT 被过滤掉了（理论上不应该发生），使用第一个
            new_gt_idx = 0
        return {
            "video_path": video_path,
            "gt_start": gt_start,
            "gt_index": new_gt_idx,
            "segments": filtered_segments,  # [(start_sec, end_sec), ...] 最多 max_segments 个
            "relevance": torch.tensor(filtered_relevance, dtype=torch.float32),
        }

class VGGSOundDatasetCrossRetrieval(Dataset):
    def __init__(
        self,
        videos_source="/data1/kaisi/datasets/VGGSound/video_raw",
        ref_source="/data1/kaisi/datasets/VGGSound/vggsound.csv",
        window_size=2.0,
        stride=0.5,
        max_segments=20,  # 新增：限制每个样本的最大 segment 数量
    ):
        self.videos_source = videos_source
        self.ref_source = ref_source
        self.window_size = float(window_size)
        self.stride = float(stride)
        self.max_segments = int(max_segments)  # 最多保留 10 个 segments
        self.video_list = self._collect_mp4_from_csv()
        self._duration_cache = {}

    def __len__(self):
        return len(self.video_list)

    def _collect_mp4_from_csv(self):
        """Collect mp4 files from CSV file (format: video_id,start_time,label,split)."""
        video_paths = []
        with open(self.ref_source, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    video_id = parts[0]
                    start_time = parts[1]
                    # Video filename format: {video_id}_{start_time:06d}.mp4
                    video_filename = f"{video_id}_{int(start_time):06d}.mp4"
                    video_path = os.path.join(self.videos_source, video_filename)
                    if os.path.exists(video_path):
                        video_paths.append(video_path)
        return video_paths

    def _get_video_duration(self, video_path):
        """通过元数据估计时长，失败时回退 10s。"""
        if video_path in self._duration_cache:
            return self._duration_cache[video_path]
        try:
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(video_path)
            duration = decoder.metadata.duration_seconds
        except Exception:
            duration = 10.0
        self._duration_cache[video_path] = float(duration)
        return self._duration_cache[video_path]

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        duration = self._get_video_duration(video_path)

        # 动态生成切片起点
        if duration <= self.window_size:
            starts = [0.0]
        else:
            starts = []
            cur = 0.0
            limit = duration - self.window_size
            while cur <= limit + 1e-6:
                starts.append(round(cur, 4))
                cur += self.stride
            if not starts:
                starts = [0.0]

        segments = [(s, min(s + self.window_size, duration)) for s in starts]

        # 随机选 GT 段（使用按样本固定的随机数生成器，避免污染全局随机状态）
        rng = np.random.default_rng(42 + idx)
        gt_idx = int(rng.integers(len(segments)))
        gt_start = segments[gt_idx][0]

        relevance = []
        for i, (s, _) in enumerate(segments):
            if i == gt_idx:
                relevance.append(2)
            elif abs(s - gt_start) <= 0.5:
                relevance.append(1)
            else:
                relevance.append(0)

        # 筛选 segments：保留 relevance=1 和 relevance=2 的，然后从 relevance=0 中采样到总数 max_segments
        relevance_array = np.array(relevance)
        high_relevance_indices = np.where(relevance_array >= 1)[0]  # relevance=1 和 2 的索引
        low_relevance_indices = np.where(relevance_array == 0)[0]  # relevance=0 的索引
        
        # 计算需要从 relevance=0 中采样多少个
        num_high_relevance = len(high_relevance_indices)
        num_to_sample = max(0, self.max_segments - num_high_relevance)
        
        # 如果 relevance=0 的数量足够，随机采样；否则全部保留
        if len(low_relevance_indices) > 0 and num_to_sample > 0:
            if len(low_relevance_indices) <= num_to_sample:
                selected_low_indices = low_relevance_indices
            else:
                # 使用同一个按样本固定的 RNG 保证可复现
                selected_low_indices = rng.choice(
                    low_relevance_indices, 
                    size=num_to_sample, 
                    replace=False
                )
            selected_indices = np.concatenate([high_relevance_indices, selected_low_indices])
        else:
            # 如果没有 relevance=0 的，或者不需要采样，只保留高 relevance 的
            selected_indices = high_relevance_indices
        
        # 排序保持顺序
        selected_indices = np.sort(selected_indices)
        
        # 重新构建 segments 和 relevance
        filtered_segments = [segments[i] for i in selected_indices]
        filtered_relevance = [relevance[i] for i in selected_indices]
        
        # 更新 gt_index（因为 segments 列表变了）
        new_gt_idx = np.where(selected_indices == gt_idx)[0]
        if len(new_gt_idx) > 0:
            new_gt_idx = int(new_gt_idx[0])
        else:
            # 如果 GT 被过滤掉了（理论上不应该发生），使用第一个
            new_gt_idx = 0
        return {
            "video_path": video_path,
            "gt_start": gt_start,
            "gt_index": new_gt_idx,
            "segments": filtered_segments,  # [(start_sec, end_sec), ...] 最多 max_segments 个
            "relevance": torch.tensor(filtered_relevance, dtype=torch.float32),
        }

if __name__ == "__main__":  
    # dataset = AvsyncDatasetCrossRetrieval()
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    # for batch in dataloader:
    #     print(batch)
    #     print(batch['segments'])
    #     break

    dataset = VGGSOundDatasetCrossRetrieval()
    print(len(dataset))
    print(dataset[0])

