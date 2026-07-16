import json
import os

root = r'f:\AI_projekt\Code\airfoil-ai'
os.chdir(root)

notebooks = {
    'notebooks/01_explore_airfrans.ipynb': [
        ('Tlakove pole — alpha={meta[\'angle_of_attack\']:.1f}°, ', 'Pressure field — alpha={meta[\'angle_of_attack\']:.1f}°, '),
        ('Magnituda rychlosti |U| — alpha={meta[\'angle_of_attack\']:.1f}°', 'Velocity magnitude |U| — alpha={meta[\'angle_of_attack\']:.1f}°'),
        ('# Sloupec 0: spolecna skala v Pa (co mas teď)', '# Column 0: common scale in Pa (your current value)'),
        ('# Sloupec 2: spolecna skala v Cp', '# Column 2: common scale in Cp'),
        ('Range uhlu nabehu: [-4.9°, 14.9°]', 'AoA range: [-4.9°, 14.9°]'),
        ('Range rychlosti:    [{velocities.min():.1f}, {velocities.max():.1f}] m/s', 'Velocity range:    [{velocities.min():.1f}, {velocities.max():.1f}] m/s'),
        ('Vstupni rychlost [m/s]', 'Inlet velocity [m/s]'),
        ('Uhel nabehu [stupne]', 'Angle of attack [degrees]'),
        ('Tlak [Pa]', 'Pressure [Pa]'),
    ],
    'notebooks/02_preprocess_to_grid.ipynb': [
        ('# --- Konfigurace ---', '# --- Configuration ---'),
        ('GRID_SIZE = 128                      # rozliseni mrizky (128x128)', 'GRID_SIZE = 128                      # grid resolution (128x128)'),
        ('DOMAIN = (-1.0, 3.0, -1.5, 1.5)      # (x_min, x_max, y_min, y_max) v metre', 'DOMAIN = (-1.0, 3.0, -1.5, 1.5)      # (x_min, x_max, y_min, y_max) in meters'),
        ('for i in tqdm(range(N), desc="Interpolace")', 'for i in tqdm(range(N), desc="Interpolation")'),
        ('print(f"\\nHotovo za {elapsed:.1f}s ({elapsed/N*1000:.0f} ms/simulace)")', 'print(f"\\nDone in {elapsed:.1f}s ({elapsed/N*1000:.0f} ms/sample)")'),
        ('# Kontrola NaN', '# NaN check'),
        ('print(f"NaN ve vstupech: {n_nan_X}")', 'print(f"NaN in inputs: {n_nan_X}")'),
        ('print(f"NaN ve vystupech: {n_nan_Y}")', 'print(f"NaN in outputs: {n_nan_Y}")'),
        ('# Pokud jsou NaN, nahradime nulou', '# If there are NaNs, replace them with zero'),
        ('print("  -> NaN ve vstupech nahrazeny nulou")', 'print("  -> NaN in inputs replaced with zero")'),
        ('print("  -> NaN ve vystupech nahrazeny nulou")', 'print("  -> NaN in outputs replaced with zero")'),
        ('# Rozsahy jednotlivych kanalu', '# Ranges of individual channels'),
        ('print("\\nRozsahy vstupnich kanalu:")', 'print("\\nInput channel ranges:")'),
        ('print("Rozsahy vystupnich kanalu:")', 'print("Output channel ranges:")'),
        ('for i, name in enumerate(["SDF", "uhel", "rychlost"]):', 'for i, name in enumerate(["SDF", "angle", "velocity"]):'),
        ('for i, name in enumerate(["tlak", "u", "v"]):', 'for i, name in enumerate(["pressure", "u", "v"]):'),
        ('# Normalizace per kanal. Statistiky pocitame pres vsechny vzorky a pixely (osy 0,2,3)', '# Normalize per channel. Statistics are computed over all samples and pixels (axes 0,2,3)'),
        ('# Pridame male epsilon aby nedoslo k deleni nulou', '# Add a small epsilon to avoid division by zero'),
        ('print("Po normalizaci (melo by byt ~0 mean, ~1 std):")', 'print("After normalization (it should be ~0 mean, ~1 std):")'),
        ('print("\\nUlozene statistiky vystupu (pro denormalizaci pri inferenci):")', 'print("\\nSaved output statistics (for denormalization during inference):")'),
        ('for i, name in enumerate(["tlak", "u", "v"]):', 'for i, name in enumerate(["pressure", "u", "v"]):'),
        ('# Nacteni zpatky', '# Load back'),
        ('# Vizualizace nekolika nahodnych vzorku (denormalizovany tlak)', '# Visualize a few random samples (denormalized pressure)'),
        ('# Denormalizace tlaku zpatky do Pa', '# Denormalize pressure back to Pa'),
        ('ax.set_title(f"Vzorek {idx}: tlak [Pa]")', 'ax.set_title(f"Sample {idx}: pressure [Pa]")'),
        ('print("\\nPreprocessing hotovy! Data jsou pripravena k treninku.")', 'print("\\nPreprocessing complete! Data are ready for training.")'),
    ],
    'notebooks/03_train.ipynb': [
        ('X = data["X"]  # (800, 3, 128, 128) - vstupy', 'X = data["X"]  # (800, 3, 128, 128) - inputs'),
        ('Y = data["Y"]  # (800, 3, 128, 128) - vystupy', 'Y = data["Y"]  # (800, 3, 128, 128) - outputs'),
        ('# Statistiky pro pozdejsi denormalizaci', '# Statistics for later denormalization'),
        ('print(f"Dataset: {len(full_dataset)} vzorku")', 'print(f"Dataset: {len(full_dataset)} samples")'),
        ('# Test - jeden vzorek', '# Test - one sample'),
        ('print(f"Jeden vstup: {x_sample.shape}")', 'print(f"One input: {x_sample.shape}")'),
        ('print(f"Jeden vystup: {y_sample.shape}")', 'print(f"One output: {y_sample.shape}")'),
        ('# Nacti nejlepsi model', '# Load the best model'),
        ('# Vezmi par validacnich vzorku', '# Take a few validation samples'),
        ('# Spolecna skala', '# Shared scale'),
        ('axes[row, 0].set_title(f"Vzorek {idx}: GROUND TRUTH (tlak)")', 'axes[row, 0].set_title(f"Sample {idx}: GROUND TRUTH (pressure)")'),
        ('axes[row, 1].set_title("PREDIKCE")', 'axes[row, 1].set_title("PREDICTION")'),
        ('axes[row, 2].set_title("ROZDIL (pred - true)")', 'axes[row, 2].set_title("DIFFERENCE (pred - true)")'),
        ('print("Vlevo: skutecnost | Uprostred: predikce | Vpravo: chyba")', 'print("Left: ground truth | Center: prediction | Right: error")'),
    ],
    'notebooks/04_evaluate.ipynb': [
        ('print("\\nDenormalizacni statistiky (vystup):")', 'print("\\nDenormalization statistics (output):")'),
        ('def denorm_Y(arr_norm):', 'def denorm_Y(arr_norm):'),
        ('"""Z normalizovaneho na fyzikalni jednotky. arr_norm: (..., 3, H, W)"""', '"""From normalized to physical units. arr_norm: (..., 3, H, W)"""'),
        ('print(f"Predikce: {preds.shape}")', 'print(f"Prediction: {preds.shape}")'),
        ('# Denormalizace do fyzikalnich jednotek', '# Denormalize to physical units'),
        ('print("Denormalizovano do Pa / m/s.")', 'print("Denormalized to Pa / m/s.")'),
        ('"""MAE, RMSE, R2 pro jedno pole (flattened)."""', '"""MAE, RMSE, R2 for a single field (flattened)."""'),
        ("print(f\"{'Kanal':<10}{'MAE':>12}{'RMSE':>12}{'R2':>10}\")", "print(f\"{'Channel':<10}{'MAE':>12}{'RMSE':>12}{'R2':>10}\")"),
        ('for i, (name, unit) in enumerate([("tlak", "Pa"), ("u", "m/s"), ("v", "m/s")]):', 'for i, (name, unit) in enumerate([("pressure", "Pa"), ("u", "m/s"), ("v", "m/s")]):'),
        ('print(f"Tlak na povrchu (pixely s SDF < {surface_threshold} m):")', 'print(f"Pressure on the surface (pixels with SDF < {surface_threshold} m):")'),
        ('print(f"  Pocet povrchovych pixelu: {surface_mask.sum():,}")', 'print(f"  Number of surface pixels: {surface_mask.sum():,}")'),
        ('print("\\nPozn.: Surface metriky byvaji horsi nez volume - u povrchu jsou nejvetsi")', 'print("\\nNote: Surface metrics are often worse than volume metrics, because the surface has the strongest gradients")'),
        ('print("gradienty a U-Net ma tendenci je vyhlazovat.")', 'print("and U-Net tends to smooth them out.")'),
        ('"""Odhadne Cl integraci tlaku kolem povrchu profilu na gridu."""', '"""Estimate Cl by integrating pressure around the airfoil surface on the grid."""'),
        ('pressure_field: (128, 128) tlak v Pa', 'pressure_field: (128, 128) pressure in Pa'),
        ('angle_rad:      uhel nabehu v radianech', 'angle_rad:      angle of attack in radians'),
        ('velocity:       rychlost proudu m/s', 'velocity:       freestream velocity in m/s'),
        ('Vraci: Cl (lift koeficient, bezrozmerny)', 'Returns: Cl (lift coefficient, dimensionless)'),
        ('# Rotace do souradnic proudu: lift je kolmo na proud', '# Rotate to flow coordinates: lift is perpendicular to the flow'),
        ('# Proud miri pod uhlem angle_rad', '# The flow points at the angle angle_rad'),
        ('# Bezrozmerny koeficient: Cl = L / (0.5 * rho * V^2 * chord)', '# Dimensionless coefficient: Cl = L / (0.5 * rho * V^2 * chord)'),
        ('# chord = 1 m (charakteristicka delka)', '# chord = 1 m (characteristic length)'),
        ('angle = X_phys[idx, 1, 0, 0]      # uhel (kanal 1, konstanta)', 'angle = X_phys[idx, 1, 0, 0]      # angle (channel 1, constant)'),
        ('vel = X_phys[idx, 2, 0, 0]        # rychlost (kanal 2, konstanta)', 'vel = X_phys[idx, 2, 0, 0]        # velocity (channel 2, constant)'),
        ('print(f"  Cl (predikce):     {cl_pred:.4f}")', 'print(f"  Cl (prediction):     {cl_pred:.4f}")'),
        ('ax.set_title("Parity plot: Cl predikce vs skutecnost")', 'ax.set_title("Parity plot: Cl prediction vs ground truth")'),
        ('ax.set_xlabel("Cl skutecne (z ground truth)")', 'ax.set_xlabel("Cl ground truth")'),
        ('ax.set_ylabel("Cl predikovane")', 'ax.set_ylabel("Cl predicted")'),
        ('print("VYBORNE - model radi profily podle vztlaku temer dokonale")', 'print("EXCELLENT - the model ranks airfoils by lift almost perfectly")'),
        ('print("DOBRE - model dobre zachycuje trend vztlaku")', 'print("GOOD - the model captures the lift trend well")'),
        ('print("OK - trend je zachycen, ale s chybami")', 'print("OK - the trend is captured, but with some error")'),
        ('print("SLABE - model nezachycuje poradi spolehlive")', 'print("WEAK - the model does not reliably capture the ranking")'),
        ('print(f"\\nPrumerna relativni chyba Cl: {rel_err:.1f}%")', 'print(f"\\nAverage relative Cl error: {rel_err:.1f}%")'),
        ('# Chyba na vzorek (MSE v poli tlaku)', '# Error per sample (MSE in the pressure field)'),
        ('# Uhly nabehu vsech vzorku', '# Angles of attack of all samples'),
        ('axes[0].set_title("Chyba vs uhel nabehu")', 'axes[0].set_title("Error vs angle of attack")'),
        ('axes[1].set_xlabel("Rychlost [m/s]")', 'axes[1].set_xlabel("Velocity [m/s]")'),
        ('axes[1].set_title("Chyba vs rychlost")', 'axes[1].set_title("Error vs velocity")'),
        ('print(f"Korelace chyby s rychlosti:    {np.corrcoef(velocities, per_sample_err)[0,1]:.3f}")', 'print(f"Correlation of error with velocity:    {np.corrcoef(velocities, per_sample_err)[0,1]:.3f}")'),
        ('print("\\nVyssi rychlost casto = vetsi tlaky = vetsi absolutni chyba (ocekavane).")', 'print("\\nHigher velocity often means larger pressures and therefore larger absolute error (expected).")'),
        ('"""Vytvori masku tvaru a spocita SDF, ve stejnem formatu jako trenovaci data."""', '"""Create a shape mask and compute the SDF in the same format as the training data."""'),
        ('cx, cy = 48, 64   # stred tvaru v pixelech', 'cx, cy = 48, 64   # shape center in pixels'),
        ('# Trojuhelnik', '# Triangle'),
        ('# SDF: vzdalenost od tvaru (vne kladna)', '# SDF: distance from the shape (positive outside)'),
    ],
    'notebooks/05_Xfoil_library_explore.ipynb': [
        ('# ← souřadnice PŘIJDOU z argumentu', '# ← coordinates come from the argument'),
        ('# jak to vidí XFOIL', '# how XFOIL sees it'),
        ('# jak to vidí notebook', '# how the notebook sees it'),
        ('# ← XFOILu krátký název', '# ← short name for XFOIL'),
        ('# --- test obou větví ---', '# --- test both branches ---'),
    ],
    'notebooks/06_testing_airfoil_generator.ipynb': [
        ('Hledané Cl:    1.000', 'Target Cl:    1.000'),
        ('Nalezené Cl:   1.000', 'Found Cl:   1.000'),
        ('Počet vyhodnocení: 384', 'Number of evaluations: 384'),
        ('    Vrací JEDNO číslo (menší = lepší).', '    Returns ONE value (smaller = better).'),
        ('    Tady: kvadratická chyba mezi predikovaným a cílovým Cl.', '    Here: quadratic error between predicted and target Cl.'),
        ('    Cl = target_cl  ->  chyba 0 (ideál).', '    Cl = target_cl  -> error 0 (ideal).'),
        ('    Najde NACA parametry [m, p, t], které dají zadané cílové Cl.', '    Finds NACA parameters [m, p, t] that yield the requested target Cl.'),
        ('print(f"Hledané Cl:    1.000")', 'print(f"Target Cl:    1.000")'),
        ('print(f"Nalezené Cl:   {out[\'cl\']:.3f}")', 'print(f"Found Cl:   {out[\'cl\']:.3f}")'),
        ('print(f"Počet vyhodnocení: {out[\'n_evals\']}")', 'print(f"Number of evaluations: {out[\'n_evals\']}")'),
    ],
}

for rel_path, replacements in notebooks.items():
    path = os.path.join(root, rel_path)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') not in {'code', 'markdown'}:
            continue
        src = cell.get('source', [])
        if isinstance(src, list):
            text = ''.join(src)
            new_text = text
            for old, new in replacements:
                new_text = new_text.replace(old, new)
            if new_text != text:
                cell['source'] = new_text.splitlines(keepends=True)
                changed = True
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write('\n')

print('Notebook translations completed.')
