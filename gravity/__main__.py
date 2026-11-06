import torch

from transformers import VJEPA2Model, VJEPA2VideoProcessor, VJEPA2Config
from torch.utils.data import DataLoader

import metrics
from datasets import AVoEMicroDataset

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset = AVoEMicroDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    config = VJEPA2Config.from_pretrained(MODEL_ID)
    config.output_hidden_states = True
    config.output_attention_weights = True

    model = VJEPA2Model.from_pretrained(MODEL_ID, config=config).to(device)
    processor = VJEPA2VideoProcessor.from_pretrained(MODEL_ID)

    for videos in dataloader:
        videos = processor(videos).data["pixel_values_videos"]
        videos = torch.stack(videos).to(device)
        print(
            surprise := metrics.surprise(
                model, videos, n_context_frames=12, n_future_frames=12
            )
        )
        break
