"""Geometrie — generování NACA 4-digit profilů a převod na SDF grid.

NACA 4-digit profil je definovaný třemi čísly:
  m = maximální prohnutí (camber) v % tětivy
  p = pozice maximálního prohnutí v desetinách tětivy
  t = maximální tloušťka v % tětivy

Např. NACA 2412: m=2%, p=40%, t=12%
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# Doména gridu (stejná jako v tréninkových datech)
DOMAIN = (-1.0, 3.0, -1.5, 1.5)
GRID_SIZE = 128


def naca4_coordinates(m, p, t, n_points=200):
    """Vrátí (x, y) souřadnice obrysu NACA 4-digit profilu.

    m, p, t jsou v desetinné formě (m=0.02 pro 2%, p=0.4, t=0.12).
    Vrací uzavřený obrys: horní strana zezadu dopředu, dolní strana dopředu dozadu.
    """
    # Kosinové rozložení bodů (hustší u náběžné/odtokové hrany)
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2

    # Tloušťková distribuce (symetrický profil)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)

    # Střední čára (camber line)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if p > 0 and m > 0:
        # Před maximem prohnutí
        mask1 = x < p
        yc[mask1] = m / p**2 * (2 * p * x[mask1] - x[mask1]**2)
        dyc[mask1] = 2 * m / p**2 * (p - x[mask1])
        # Za maximem prohnutí
        mask2 = x >= p
        yc[mask2] = m / (1 - p)**2 * ((1 - 2*p) + 2*p*x[mask2] - x[mask2]**2)
        dyc[mask2] = 2 * m / (1 - p)**2 * (p - x[mask2])

    theta = np.arctan(dyc)

    # Horní a dolní povrch (kolmo na camber line)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Uzavřený obrys: horní zezadu dopředu + dolní dopředu dozadu
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])

    return x_coords, y_coords


def rotate_airfoil(x, y, angle_deg):
    """Otočí profil o úhel náběhu (kolem počátku, tj. náběžné hrany)."""
    angle_rad = np.radians(-angle_deg)  # záporné: kladný AoA = nos nahoru
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot


def polygon_to_mask(x_coords, y_coords, domain=DOMAIN, grid_size=GRID_SIZE):
    """Převede polygon (obrys profilu) na binární masku na gridu.

    Vrací (grid_size, grid_size) bool array — True uvnitř profilu.
    """
    from matplotlib.path import Path

    x_min, x_max, y_min, y_max = domain
    # Pravidelná mřížka
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    points = np.column_stack([gx.ravel(), gy.ravel()])

    # Polygon test: které body mřížky jsou uvnitř obrysu
    polygon = Path(np.column_stack([x_coords, y_coords]))
    mask = polygon.contains_points(points).reshape(grid_size, grid_size)
    return mask


def mask_to_sdf(mask, domain=DOMAIN, grid_size=GRID_SIZE):
    """Spočítá signed distance function z binární masky.

    Vrací SDF kde hodnota = vzdálenost od povrchu profilu (kladná vně).
    Měřítko odpovídá tréninkovým datům (vzdálenost v metrech).
    """
    # Vzdálenost od profilu (vně profilu)
    sdf = distance_transform_edt(~mask).astype(np.float32)

    # Převod z pixelů na metry (podle velikosti domény)
    x_min, x_max, y_min, y_max = domain
    dx = (x_max - x_min) / grid_size
    sdf = sdf * dx

    return sdf


def naca_to_sdf(m, p, t, angle_deg=0, domain=DOMAIN, grid_size=GRID_SIZE):
    """Kompletní pipeline: NACA parametry → SDF grid.

    m, p, t v desetinné formě. angle_deg ve stupních.
    Vrací (sdf, mask).
    """
    x, y = naca4_coordinates(m, p, t)
    x, y = rotate_airfoil(x, y, angle_deg)
    mask = polygon_to_mask(x, y, domain, grid_size)
    sdf = mask_to_sdf(mask, domain, grid_size)
    return sdf, mask


def coordinates_to_sdf(x, y, angle_deg=0, domain=DOMAIN, grid_size=GRID_SIZE):
    """Pipeline pro vlastní souřadnice (z .dat souboru): coords → SDF grid.

    x, y jsou souřadnice obrysu profilu (tětiva normalizovaná na ~0-1).
    """
    x, y = rotate_airfoil(np.asarray(x), np.asarray(y), angle_deg)
    mask = polygon_to_mask(x, y, domain, grid_size)
    sdf = mask_to_sdf(mask, domain, grid_size)
    return sdf, mask


def parse_dat_file(content):
    """Zparsuje .dat soubor (Selig/UIUC formát) na (x, y) souřadnice.

    content: string obsah souboru.
    Vrací (x, y) numpy arrays.
    """
    lines = content.strip().split("\n")
    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                x_val, y_val = float(parts[0]), float(parts[1])
                # Filtruj hlavičku (např. počet bodů bývá > 1)
                if -2 <= x_val <= 2 and -2 <= y_val <= 2:
                    coords.append((x_val, y_val))
            except ValueError:
                continue  # přeskoč nečíselné řádky (jméno profilu)

    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]


def canvas_to_sdf(json_data, canvas_size=400, domain=DOMAIN, grid_size=GRID_SIZE):
    """Převede nakreslený tvar z streamlit-drawable-canvas na SDF grid.

    json_data: canvas.json_data (objekty nakreslené uživatelem)
    canvas_size: velikost canvasu v pixelech (pro normalizaci souřadnic)

    Vrací (sdf, mask) nebo (None, None) pokud nic platného není nakresleno.
    """
    from matplotlib.path import Path

    if json_data is None or "objects" not in json_data:
        return None, None
    objects = json_data["objects"]
    if len(objects) == 0:
        return None, None

    # Vezmeme první nakreslený objekt (path nebo polygon)
    obj = objects[0]

    points = []
    if obj["type"] == "path":
        # Freedraw: path je seznam SVG příkazů [['M', x, y], ['L', x, y], ['Q', ...], ...]
        for cmd in obj["path"]:
            if len(cmd) >= 3 and isinstance(cmd[1], (int, float)):
                points.append((cmd[1], cmd[2]))
    elif obj["type"] in ("polygon", "rect", "circle"):
        # Pro jiné typy zkusíme bounding box / vrcholy (zjednodušeně)
        if "points" in obj:
            for pt in obj["points"]:
                points.append((pt["x"], pt["y"]))

    if len(points) < 3:
        return None, None

    points = np.array(points, dtype=float)

    # Canvas souřadnice (y dolů) → normalizované [0,1] s y nahoru
    px = points[:, 0] / canvas_size
    py = 1.0 - points[:, 1] / canvas_size  # flip y

    # Namapuj na typickou airfoil pozici v doméně:
    # x: profil bývá kolem [0,1] v doméně [-1,3]
    # y: kolem 0 v doméně [-1.5,1.5]
    # Vycentrujeme nakreslený tvar a zmenšíme na rozumnou velikost (~chord 1)
    px = px - px.mean()
    py = py - py.mean()
    scale = 1.0 / (px.max() - px.min() + 1e-8)  # normalizuj šířku na ~1
    px = px * scale + 0.5   # střed kolem x=0.5
    py = py * scale         # střed kolem y=0

    # Rasterizace polygonu na grid
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
