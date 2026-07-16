"""Inference - building inputs, running the model, denormalizing outputs, and computing Cl."""

import numpy as np
import torch

GRID_SIZE = 128
DOMAIN = (-1.0, 3.0, -1.5, 1.5)


def build_input(sdf, angle_deg, velocity, norm_stats, grid_size=GRID_SIZE):
    angle_rad = np.radians(angle_deg)
    inp = np.stack([
        sdf,
        np.full((grid_size, grid_size), angle_rad, dtype=np.float32),
        np.full((grid_size, grid_size), float(velocity), dtype=np.float32),
    ])

    X_mean = norm_stats["X_mean"][0]
    X_std = norm_stats["X_std"][0]
    inp_norm = (inp - X_mean) / (X_std + 1e-8)

    return torch.from_numpy(inp_norm[None].astype(np.float32))


def predict(model, input_tensor, norm_stats, device="cpu"):
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor.to(device)).cpu().numpy()[0]

    Y_mean = norm_stats["Y_mean"][0]
    Y_std = norm_stats["Y_std"][0]
    pred_phys = pred_norm * Y_std + Y_mean

    pressure = pred_phys[0]
    u = pred_phys[1]
    v = pred_phys[2]
    return pressure, u, v


def compute_cl(pressure, sdf, angle_deg, velocity, rho=1.225, surface_band=0.04):
    angle_rad = np.radians(angle_deg)

    gy, gx = np.gradient(sdf)
    grad_mag = np.sqrt(gx**2 + gy**2) + 1e-8
    nx = gx / grad_mag
    ny = gy / grad_mag

    band = (sdf >= 0) & (sdf < surface_band)

    x_min, x_max, y_min, y_max = DOMAIN
    dx = (x_max - x_min) / GRID_SIZE
    dy = (y_max - y_min) / GRID_SIZE
    cell = dx * dy

    fx = -np.sum(pressure[band] * nx[band]) * cell / surface_band
    fy = -np.sum(pressure[band] * ny[band]) * cell / surface_band

    lift = -fx * np.sin(angle_rad) + fy * np.cos(angle_rad)
    q_inf = 0.5 * rho * velocity**2
    cl = lift / (q_inf * 1.0)
    return cl


def distance_from_training(sdf, ref_sdfs=None):
    mask = sdf < 0.02
    if mask.sum() < 5:
        return 1.0

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    height = np.sum(rows)
    width = np.sum(cols)

    aspect = height / (width + 1e-8)
    ood_score = min(1.0, max(0.0, (aspect - 0.15) / 0.5))
    return ood_score
