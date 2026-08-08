# coding = utf-8
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple
from transformers.models.auto import AutoModel
from .configuration_pe_audio_video import PeAudioVideoConfig
from .modeling_pe_audio import PeAudioEncoder, PeAudioContrastiveHead
from .modeling_pe_video import PeVideoEncoder, PeVideoContrastiveHead

@dataclass
class PeAudioVideoOutput(ModelOutput):
    # embeddings
    audio_embeds: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    audio_frame_embeds: Optional[torch.FloatTensor] = None
    video_frame_embeds: Optional[torch.FloatTensor] = None

class PeAudioVideoPreTrainedModel(PreTrainedModel):
    config: PeAudioVideoConfig
    supports_gradient_checkpointing = True

class PeAudioVideoModel(PeAudioVideoPreTrainedModel):
    all_tied_weights_keys = {}
    def __init__(self, config: PeAudioVideoConfig):
        audio_config = config.audio_video_config.audio_config
        video_config = config.audio_video_config.video_config
        super().__init__(config)
        self.audio_encoder = PeAudioEncoder(audio_config)
        self.video_encoder = PeVideoEncoder(video_config)
        self.video_head = PeVideoContrastiveHead(video_config.hidden_size, config.text_config.hidden_size)
        self.audio_head= PeAudioContrastiveHead(audio_config.hidden_size, config.text_config.hidden_size)
        self.audio_encoder.gradient_checkpointing = True
        self.video_encoder.gradient_checkpointing = True
    
    def _align_video_hidden_state(
        self,
        video_hidden_state: torch.Tensor,
        audio_hidden_state: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Align video_hidden_state to audio_hidden_state by nearest neighbor interpolation.
        """
        if video_hidden_state.shape[1] == audio_hidden_state.shape[1]:
            return video_hidden_state

        if padding_mask_videos is not None:
            video_lengths = padding_mask_videos.sum(dim=-1)
        else:
            video_lengths = video_hidden_state.shape[1] * video_hidden_state.new_ones(
                video_hidden_state.shape[0], dtype=torch.long
            )

        if padding_mask is not None:
            audio_lengths = padding_mask.sum(dim=-1)
        else:
            audio_lengths = audio_hidden_state.shape[1] * audio_hidden_state.new_ones(
                audio_hidden_state.shape[0], dtype=torch.long
            )
        if (audio_lengths == video_hidden_state.shape[1]).all() or (
            video_lengths == audio_hidden_state.shape[1]
        ).all():
            # no need to align taking into account the padding masks
            # note: when one of the above is true, we can expect the other to be true as there is no reason
            # to have masked audio without masked video and vice versa

            return nn.functional.interpolate(video_hidden_state, size=audio_hidden_state.shape[1], mode="nearest")

        aligned_shape = (*audio_hidden_state.shape[:2], video_hidden_state.shape[-1])
        
        aligned_hidden_state = audio_hidden_state.new_zeros(aligned_shape)

        # Convert lengths to python list once to avoid repeated .item() calls in loop
        # This is more torch.compile friendly as it moves the graph break outside the loop
        video_lengths_list = video_lengths.tolist() if torch.is_tensor(video_lengths) else list(video_lengths)
        audio_lengths_list = audio_lengths.tolist() if torch.is_tensor(audio_lengths) else list(audio_lengths)

        for i, (hidden_state, v_len, a_len) in enumerate(
            zip(video_hidden_state, video_lengths_list, audio_lengths_list)
        ):
            v_len = int(v_len)
            a_len = int(a_len)

            hidden_state = hidden_state[:v_len]
            if hidden_state.numel() > 0 and a_len > 0:
                # interpolate 的 size 需要 int 或 tuple[int]，不能是 Tensor
                interpolated_hidden_state = nn.functional.interpolate(
                    hidden_state[None].transpose(1, 2), size=a_len, mode="nearest"
                ).transpose(1, 2)[0]
                aligned_hidden_state[i, :a_len, :] = interpolated_hidden_state

        return aligned_hidden_state

    @can_return_tuple
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        return_loss=False,
        **kwargs,
    ) -> PeAudioVideoOutput:
        

        # 1. Audio Encoding
        audio_outputs = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
        )
        
        video_outputs = self.video_encoder(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
        )
        video_hidden_state = video_outputs.last_hidden_state
        audio_hidden_state = audio_outputs.last_hidden_state
        # 2. Align video hidden states to audio hidden states
        aligned_video_hidden_state = self._align_video_hidden_state(
            video_hidden_state=video_hidden_state,
            audio_hidden_state=audio_hidden_state,
            padding_mask_videos=video_outputs.padding_mask if getattr(video_outputs, "padding_mask", None) is not None else padding_mask_videos,
            padding_mask=audio_outputs.output_mask if getattr(audio_outputs, "output_mask", None) is not None else padding_mask,
        )
        
        audio_embeds = self.audio_head(audio_outputs.pooler_output)
        video_embeds = self.video_head(video_outputs.pooler_output)
        aligned_video_hidden_state = self.video_head(aligned_video_hidden_state)
        aligned_audio_hidden_state = self.audio_head(audio_hidden_state)


        return PeAudioVideoOutput(
            audio_embeds=audio_embeds,
            video_embeds=video_embeds,
            audio_frame_embeds=aligned_audio_hidden_state,
            video_frame_embeds=aligned_video_hidden_state,
        )

__all__ = ["PeAudioVideoModel"]
