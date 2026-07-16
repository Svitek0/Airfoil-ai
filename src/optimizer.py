import numpy as np

from .geometry import naca_to_sdf, naca4_coordinates
from .inference import build_input, predict, compute_cl


def evaluate(params, alpha, model, norm_stats, velocity=30.0, device="cpu"):
    m, p, t = params

    # 1) NACA parameters -> SDF grid (geometry.py)
    #    Note: AoA is passed to build_input as a condition, not to the geometry.
    #    The model was trained with a flat airfoil plus angle as an input channel,
    #    so angle_deg=0 is used here and the real angle is passed below to build_input.
    sdf, mask = naca_to_sdf(m, p, t, angle_deg=0)

    # 2) SDF + conditions -> normalized input tensor (inference.py)
    input_tensor = build_input(sdf, alpha, velocity, norm_stats)

    # 3) Prediction -> pressure, u, v in physical units
    pressure, u, v = predict(model, input_tensor, norm_stats, device=device)

    # 4) Pressure -> Cl
    cl = compute_cl(pressure, sdf, alpha, velocity)

    return {"cl": cl, "pressure": pressure, "u": u, "v": v}