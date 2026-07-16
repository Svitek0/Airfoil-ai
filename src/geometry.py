"""Geometry - generating NACA 4-digit airfoils and converting them to an SDF grid.

A NACA 4-digit airfoil is defined by three numbers:
  m = maximum camber as a percentage of chord
  p = position of maximum camber as a fraction of chord
  t = maximum thickness as a percentage of chord

Example: NACA 2412 -> m=2%, p=40%, t=12%
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


DOMAIN = (-1.0, 3.0, -1.5, 1.5)
GRID_SIZE = 128


def naca4_coordinates(m, p, t, n_points=200):
    """Return (x, y) coordinates of the NACA 4-digit airfoil outline.

    m, p, t are given in decimal form (m=0.02 for 2%, p=0.4, t=0.12).
    Returns a closed outline: upper side from trailing edge to leading edge,
    lower side from leading edge to trailing edge.
    """
    # Cosine spacing of points (denser near the leading/trailing edge)
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if p > 0 and m > 0:
        mask1 = x < p
        yc[mask1] = m / p**2 * (2 * p * x[mask1] - x[mask1]**2)
        dyc[mask1] = 2 * m / p**2 * (p - x[mask1])

        mask2 = x >= p
        yc[mask2] = m / (1 - p)**2 * ((1 - 2*p) + 2*p*x[mask2] - x[mask2]**2)
        dyc[mask2] = 2 * m / (1 - p)**2 * (p - x[mask2])

    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])

    return x_coords, y_coords


def rotate_airfoil(x, y, angle_deg):
    """Rotate the airfoil by an angle of attack around the origin (the leading edge)."""
    angle_rad = np.radians(-angle_deg)  # negative sign: positive AoA pitches the nose upward
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot


def polygon_to_mask(x_coords, y_coords, domain=DOMAIN, grid_size=GRID_SIZE):
    """Convert a polygon (the airfoil outline) to a binary mask on the grid.

    Returns a (grid_size, grid_size) boolean array where True indicates points inside the airfoil.
    """
    from matplotlib.path import Path

    x_min, x_max, y_min, y_max = domain

    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    points = np.column_stack([gx.ravel(), gy.ravel()])

    polygon = Path(np.column_stack([x_coords, y_coords]))
    mask = polygon.contains_points(points).reshape(grid_size, grid_size)
    return mask


def mask_to_sdf(mask, domain=DOMAIN, grid_size=GRID_SIZE):
    """Compute the signed distance function from a binary mask.

    Returns an SDF where values represent the distance from the airfoil surface (positive outside).
    The scale matches the training data (distance in meters).
    """
    sdf = distance_transform_edt(~mask).astype(np.float32)

    x_min, x_max, y_min, y_max = domain
    dx = (x_max - x_min) / grid_size
    sdf = sdf * dx

    return sdf


def naca_to_sdf(m, p, t, angle_deg=0, domain=DOMAIN, grid_size=GRID_SIZE):
    """Complete pipeline: NACA parameters -> SDF grid.

    m, p, t are in decimal form. angle_deg is in degrees.
    Returns (sdf, mask).
    """
    x, y = naca4_coordinates(m, p, t)
    x, y = rotate_airfoil(x, y, angle_deg)
    mask = polygon_to_mask(x, y, domain, grid_size)
    sdf = mask_to_sdf(mask, domain, grid_size)
    return sdf, mask


def coordinates_to_sdf(x, y, angle_deg=0, domain=DOMAIN, grid_size=GRID_SIZE):
    """Pipeline for custom coordinates (from a .dat file): coords -> SDF grid.

    x and y are airfoil outline coordinates (chord normalized to about 0-1).
    """
    x, y = rotate_airfoil(np.asarray(x), np.asarray(y), angle_deg)
    mask = polygon_to_mask(x, y, domain, grid_size)
    sdf = mask_to_sdf(mask, domain, grid_size)
    return sdf, mask


def parse_dat_file(content):
    """Parse a .dat file (Selig/UIUC format) into (x, y) coordinates.

    content: the file contents as a string.
    Returns (x, y) numpy arrays.
    """
    lines = content.strip().split("\n")
    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                x_val, y_val = float(parts[0]), float(parts[1])

                if -2 <= x_val <= 2 and -2 <= y_val <= 2:
                    coords.append((x_val, y_val))
            except ValueError:
                continue

    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]


def canvas_to_sdf(json_data, canvas_size=400, domain=DOMAIN, grid_size=GRID_SIZE):
    """Convert a shape drawn in streamlit-drawable-canvas to an SDF grid.

    json_data: canvas.json_data (objects drawn by the user)
    canvas_size: canvas size in pixels (for coordinate normalization)

    Returns (sdf, mask) or (None, None) if no valid shape was drawn.
    """
    from matplotlib.path import Path

    if json_data is None or "objects" not in json_data:
        return None, None
    objects = json_data["objects"]
    if len(objects) == 0:
        return None, None

    # Use the first drawn object (path or polygon)
    obj = objects[0]

    points = []
    if obj["type"] == "path":
        for cmd in obj["path"]:
            if len(cmd) >= 3 and isinstance(cmd[1], (int, float)):
                points.append((cmd[1], cmd[2]))
    elif obj["type"] in ("polygon", "rect", "circle"):
        if "points" in obj:
            for pt in obj["points"]:
                points.append((pt["x"], pt["y"]))

    if len(points) < 3:
        return None, None

    points = np.array(points, dtype=float)

    px = points[:, 0] / canvas_size
    py = 1.0 - points[:, 1] / canvas_size

    px = px - px.mean()
    py = py - py.mean()
    scale = 1.0 / (px.max() - px.min() + 1e-8)
    px = px * scale + 0.5
    py = py * scale

    x_min, x_max, y_min, y_max = domain
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    polygon = Path(np.column_stack([px, py]))
    mask = polygon.contains_points(grid_points).reshape(grid_size, grid_size)

    if mask.sum() < 5:
        return None, None

    sdf = mask_to_sdf(mask, domain, grid_size)
    return sdf, mask
