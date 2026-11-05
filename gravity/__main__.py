import torch

from transformers import VJEPA2Model, VJEPA2VideoProcessor, VJEPA2Config
from torchcodec.decoders import VideoDecoder

import metrics

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
DEVICE = "mps"


if __name__ == "__main__":
    config = VJEPA2Config.from_pretrained(MODEL_ID)
    config.output_hidden_states = True
    config.output_attention_weights = True

    model = VJEPA2Model.from_pretrained(MODEL_ID, config=config).to(DEVICE)
    processor = VJEPA2VideoProcessor.from_pretrained(MODEL_ID)
    decoder = VideoDecoder(VIDEO_URL)

    # video = decoder.get_frames_at(np.arange(0, 64)).data
    video = decoder.get_all_frames().data
    video = processor(video, return_tensors="pt").to(model.device)
    video = video.data["pixel_values_videos"]
    reversed_video = video.flip(1)

    surprise = metrics.surprise(
        model, torch.concatenate((video, reversed_video), dim=0)
    )

    print(surprise)
