"""U-Net model - building and loading trained weights."""

import torch
import segmentation_models_pytorch as smp


def build_unet():
    """Build a U-Net with the same architecture used during training.

    encoder_weights=None - for inference we do not need ImageNet weights;
    we load our own trained weights with load_state_dict.
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3,
    )


def load_model(model_path, device="cpu"):
    """Load a trained model from a .pth file and switch it to eval mode."""
    model = build_unet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
