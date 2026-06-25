import subprocess
import numpy as np
import os

def load_dat(path):
    coords = np.loadtxt(path, skiprows=1)
    x = coords[:, 0]
    y = coords[:, 1]
    return x, y

def run_xfoil(airfoil, velocity, alpha_start=-5, alpha_end=15, alpha_step=0.5,
              chord=1.0, visc=1.5e-5, xfoil_path="./xfoil.exe", n_iter=200):
    
    polar = "./polar.txt"
    re = (velocity * chord) / (visc)
    
    if os.path.exists(polar):
        os.remove(polar)

    if isinstance(airfoil, str):
        airfoil_cmd = f"NACA {airfoil}"
    else:
        x, y = airfoil                            # ← souřadnice PŘIJDOU z argumentu
        dat_name = "temp_airfoil.dat"             # jak to vidí XFOIL
        dat_path = os.path.join(".", dat_name)   # jak to vidí notebook
        coords = np.column_stack([x, y])
        np.savetxt(dat_path, coords, header="airfoil", comments="")
        airfoil_cmd = f"LOAD {dat_name}"          # ← XFOILu krátký název
    
    polar_file = "polar.txt"

    commands = f"""{airfoil_cmd}
OPER
VISC {re}
ITER {n_iter}
PACC
{polar_file}

ASEQ {alpha_start} {alpha_end} {alpha_step}
PACC

QUIT
"""
    
    result = subprocess.run(
        [xfoil_path],
        input=commands,         
        capture_output=True,
        text=True,
        cwd=".",                 
        timeout=60                
    )
    
    data = np.loadtxt("./polar.txt", skiprows=12)
    alpha = data[:, 0]
    cl    = data[:, 1]
    cd    = data[:, 2]
    cm    = data[:, 4]
    
    return alpha, cl, cd, cm


# test
#
#alpha, cl, cd, cm = run_xfoil("2412", velocity=50)
#print(f"NACA:         points {len(alpha)}, Cl max = {cl.max():.3f}")
#
#x, y = load_dat("./NACA2412.dat")
#alpha2, cl2, cd2, cm2 = run_xfoil((x, y), velocity=50)
#print(f"cordinates file : points {len(alpha2)}, Cl max = {cl2.max():.3f}")