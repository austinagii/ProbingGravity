import torch

from torch.nn import functional as F
from transformers import VJEPA2Model, VJEPA2VideoProcessor, VJEPA2Config
from torchcodec.decoders import VideoDecoder


MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
DEVICE = "mps"


@torch.no_grad()
def calculate_surprise(
    model: VJEPA2Model,
    video: torch.Tensor,
    n_context_frames: int = 30,
    n_future_frames: int = 30,
    stride: int = 2,
) -> torch.Tensor:
    """Returns the surprise score at every `stride` frames in the video."""
    B, T, C, H, W = video.shape
    config = model.config
    device = model.device

    n_window_frames = n_context_frames + n_future_frames
    n_tokens_per_step = (H // config.patch_size) * (W // config.patch_size)
    n_context_tokens = (n_context_frames // config.tubelet_size) * n_tokens_per_step
    context_mask = (
        torch.arange(n_context_tokens, device=device).unsqueeze(0).repeat(B, 1)
    )

    n_window_tokens = (n_window_frames // config.tubelet_size) * n_tokens_per_step
    target_mask = (
        torch.arange(n_context_tokens, n_window_tokens, device=device)
        .unsqueeze(0)
        .repeat(B, 1)
    )

    scores = []
    for i in range(0, T - n_window_frames + 1, stride):
        print(f"Processing window [{i}: {i + n_window_frames}])")
        ground_truth_encoding = model.encoder(
            video[:, i : i + n_window_frames]
        ).last_hidden_state
        target = ground_truth_encoding[:, n_context_tokens:]

        context = model.encoder(video[:, i : i + n_context_frames]).last_hidden_state

        prediction = model.predictor(
            encoder_hidden_states=context,
            context_mask=[context_mask],
            target_mask=[target_mask],
        ).last_hidden_state

        embedding_size = target.shape[-1]
        prediction = F.layer_norm(prediction.float(), (embedding_size,))
        target = F.layer_norm(target.float(), (embedding_size,))
        scores.append((prediction - target).abs().mean(dim=(1, 2)))

    return torch.stack(scores, dim=1)


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

    surprise = calculate_surprise(
        model, torch.concatenate((video, reversed_video), dim=0)
    )

    print(surprise)
