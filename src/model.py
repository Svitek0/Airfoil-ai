"""U-Net model — stavba a načítání natrénovaných vah."""

import torch
import segmentation_models_pytorch as smp


def build_unet():
    """Postaví U-Net se stejnou architekturou jako při tréninku.

    encoder_weights=None — při inferenci nepotřebujeme ImageNet váhy,
    načítáme vlastní natrénované přes load_state_dict.
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,   # SDF + úhel + rychlost
        classes=3,       # tlak + u + v
    )


def load_model(model_path, device="cpu"):
    """Načte natrénovaný model z .pth souboru a přepne do eval módu."""
    model = build_unet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
