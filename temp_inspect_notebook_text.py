import json, os
path = r'f:\AI_projekt\Code\airfoil-ai\notebooks\06_testing_airfoil_generator.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    if 'Hledané' in src or 'Nalezené' in src or 'Počet' in src or 'NACA parametry' in src:
        print('CELL', i, cell.get('cell_type'))
        print(src)
        print('---')
