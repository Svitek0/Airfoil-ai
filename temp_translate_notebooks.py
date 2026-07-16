import json
import os
import glob

root = r'f:\AI_projekt\Code\airfoil-ai'
os.chdir(root)

replacements = {
    'notebooks/01_explore_airfrans.ipynb': [
        ("Tlakove pole — alpha={meta['angle_of_attack']:.1f}°, ", "Pressure field — alpha={meta['angle_of_attack']:.1f}°, "),
        ("Magnituda rychlosti |U| — alpha={meta['angle_of_attack']:.1f}°", "Velocity magnitude |U| — alpha={meta['angle_of_attack']:.1f}°"),
        ("# Sloupec 0: spolecna skala v Pa (co mas teď)", "# Column 0: common scale in Pa (your current value)"),
        ("vmin, vmax = np.percentile(speed, [1, 99])  # ← pridej tohle", "vmin, vmax = np.percentile(speed, [1, 99])  # ← add this"),
        ("sc = ax.scatter(x, y, c=speed, cmap=\"viridis\", s=1, vmin=vmin, vmax=vmax)  # ← vmin/vmax", "sc = ax.scatter(x, y, c=speed, cmap=\"viridis\", s=1, vmin=vmin, vmax=vmax)  # ← vmin/vmax"),
        ("Range uhlu nabehu: [-4.9°, 14.9°]", "AoA range: [-4.9°, 14.9°]"),
        ("print(f\"\\nRange uhlu nabehu: [{aoas.min():.1f}°, {aoas.max():.1f}°]\")", "print(f\"\\nAoA range: [{aoas.min():.1f}°, {aoas.max():.1f}°]\")"),
    ],
    'notebooks/02_preprocess_to_grid.ipynb': [],
    'notebooks/03_train.ipynb': [],
    'notebooks/04_evaluate.ipynb': [
        ("Vzorek 0: uhel=11.3°, V=36.6 m/s", "Sample 0: AoA=11.3°, V=36.6 m/s"),
        ("print(f\"Vzorek {idx}: uhel={np.degrees(angle):.1f}°, V={vel:.1f} m/s\")", "print(f\"Sample {idx}: AoA={np.degrees(angle):.1f}°, V={vel:.1f} m/s\")"),
        ("axes[0].set_xlabel(\"Uhel nabehu [°]\")", "axes[0].set_xlabel(\"Angle of attack [°]\")"),
        ("axes[0].set_ylabel(\"MSE tlaku [Pa²]\")", "axes[0].set_ylabel(\"Pressure MSE [Pa²]\")"),
        ("axes[1].set_ylabel(\"MSE tlaku [Pa²]\")", "axes[1].set_ylabel(\"Pressure MSE [Pa²]\")"),
    ],
    'notebooks/05_Xfoil_library_explore.ipynb': [
        ("# ← souřadnice PŘIJDOU z argumentu", "# ← coordinates come from the argument"),
        ("# jak to vidí XFOIL", "# how XFOIL sees it"),
        ("# jak to vidí notebook", "# how the notebook sees it"),
        ("# ← XFOILu krátký název", "# ← short name for XFOIL"),
        ("# --- test obou větví ---", "# --- test both branches ---"),
    ],
    'notebooks/06_testing_airfoil_generator.ipynb': [
        ("Hledané Cl:    1.000", "Target Cl:    1.000"),
        ("Nalezené Cl:   1.000", "Found Cl:   1.000"),
        ("Počet vyhodnocení: 384", "Number of evaluations: 384"),
        ("    Vrací JEDNO číslo (menší = lepší).", "    Returns ONE value (smaller = better)."),
        ("    Tady: kvadratická chyba mezi predikovaným a cílovým Cl.", "    Here: quadratic error between predicted and target Cl."),
        ("    Cl = target_cl  ->  chyba 0 (ideál).", "    Cl = target_cl  -> error 0 (ideal)."),
        ("    Najde NACA parametry [m, p, t], které dají zadané cílové Cl.", "    Finds NACA parameters [m, p, t] that yield the requested target Cl."),
        ("print(f\"Hledané Cl:    1.000\")", "print(f\"Target Cl:    1.000\")"),
        ("print(f\"Nalezené Cl:   {out['cl']:.3f}\")", "print(f\"Found Cl:   {out['cl']:.3f}\")"),
        ("print(f\"Počet vyhodnocení: {out['n_evals']}\")", "print(f\"Number of evaluations: {out['n_evals']}\")"),
    ],
}

for rel_path, pairs in replacements.items():
    path = os.path.join(root, rel_path)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    changed = False
    for cell in nb.get('cells', []):
        if isinstance(cell, dict) and cell.get('cell_type') in {'code', 'markdown'}:
            src = cell.get('source', [])
            if isinstance(src, list):
                text = ''.join(src)
                new_text = text
                for old, new in pairs:
                    new_text = new_text.replace(old, new)
                if new_text != text:
                    cell['source'] = new_text.splitlines(keepends=True)
                    changed = True
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write('\n')

print('Notebook translations updated.')
