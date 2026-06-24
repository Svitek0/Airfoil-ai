"""Airfoil AI — interactive surrogate model for 2D airflow around airfoils, built with Streamlit.
Run: streamlit run app.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st

from src.model import load_model
from src.geometry import (naca_to_sdf, coordinates_to_sdf, parse_dat_file,
                          canvas_to_sdf, DOMAIN)
from src.inference import build_input, predict, compute_cl, distance_from_training

st.set_page_config(
    page_title="Airfoil AI",
    page_icon="✈",
    layout="wide",
)

MODEL_PATH = "models/best_model.pth"
STATS_PATH = "models/norm_stats.npz"
DEVICE = "cpu"

VIEW = (-0.5, 2.0, -1.0, 1.0)
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em;
        color: #2d6cdf; margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.05rem; opacity: 0.75; margin-top: 0; font-weight: 400;
    }
    .metric-card {
        background: rgba(45, 108, 223, 0.08);
        border-left: 3px solid #2d6cdf;
        padding: 0.8rem 1rem; border-radius: 4px; margin: 0.3rem 0;
    }
    .metric-label { font-size: 0.85rem; opacity: 0.7; }
    .metric-value { font-size: 1.6rem; font-weight: 600; color: #2d6cdf; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, device=DEVICE)

@st.cache_resource
def get_stats():
    return dict(np.load(STATS_PATH))

try:
    model = get_model()
    norm_stats = get_stats()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

st.markdown('<p class="main-title">Airfoil AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Neural surrogate model predicting 2D airflow around airfoils. Development assisted by Claude Opus 4.8</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

if not model_loaded:
    st.error(f"Failed to load model: {load_error}")
    st.info("Check `models/best_model.pth` and `models/norm_stats.npz`.")
    st.stop()

st.sidebar.header("Input")

input_mode = st.sidebar.radio("Geometry source",
                              ["NACA generator", "Upload .dat file", "Draw shape"])

if input_mode == "NACA generator":
    st.sidebar.markdown("**NACA 4-digit parameters**")
    m = st.sidebar.slider("Max camber (m) [%]", 0, 9, 2)
    p = st.sidebar.slider("Camber position (p) [×10%]", 1, 9, 4)
    t = st.sidebar.slider("Thickness (t) [%]", 6, 24, 12)
    naca_code = f"{m}{p}{t:02d}"
    st.sidebar.markdown(f"→ **NACA {naca_code}**")
    uploaded_coords = None
elif input_mode == "Upload .dat file":
    st.sidebar.markdown("**Upload airfoil coordinates**")
    uploaded = st.sidebar.file_uploader("Selig/UIUC .dat format", type=["dat", "txt"])
    uploaded_coords = None
    if uploaded is not None:
        file_content = uploaded.read().decode("utf-8", errors="ignore")
        try:
            uploaded_coords = parse_dat_file(file_content)
            st.sidebar.success(f"Loaded {len(uploaded_coords[0])} points")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

else:
    st.sidebar.markdown("**Draw a shape**")
    st.sidebar.caption("Draw on the canvas in the main area. "
                       "Try an airfoil-like teardrop — or a triangle/square to see "
                       "the model fail on out-of-distribution shapes.")
    uploaded_coords = None

st.sidebar.markdown("**Flow conditions**")
angle = st.sidebar.slider("Angle of attack [°]", -5, 15, 5)
velocity = st.sidebar.slider("Velocity [m/s]", 20, 90, 50)

if input_mode == "Draw shape":
    from streamlit_drawable_canvas import st_canvas
    st.markdown("**Draw a closed shape** (the canvas connects your stroke automatically)")
    canvas_result = st_canvas(
        fill_color="rgba(45, 108, 223, 0.3)",
        stroke_width=3,
        stroke_color="#2d6cdf",
        background_color="#0e1117",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    sdf, mask = canvas_to_sdf(canvas_result.json_data, canvas_size=400)
    geometry_ok = sdf is not None
    if not geometry_ok:
        st.info("Draw a shape on the canvas to see the prediction.")
        st.stop()
elif input_mode == "NACA generator":
    sdf, mask = naca_to_sdf(m / 100, p / 10, t / 100, angle_deg=angle)
    geometry_ok = True
elif uploaded_coords is not None:
    sdf, mask = coordinates_to_sdf(uploaded_coords[0], uploaded_coords[1], angle_deg=angle)
    geometry_ok = True
else:
    geometry_ok = False

if not geometry_ok:
    st.info("Upload a .dat file to see the prediction.")
    st.stop()

if st.sidebar.button("AoA Sweep"):
    aoa_sweep = True
else:
    aoa_sweep = False

profile_name = "Airfoil"
if input_mode == "NACA generator":
    profile_name = f"NACA {m}{p}{t:02d}"
elif input_mode == "Upload .dat file" and uploaded is not None:
    profile_name = uploaded.name.rsplit(".",1)[0]
else:
    profile_name = "Hand-drawn shape"


input_tensor = build_input(sdf, angle, velocity, norm_stats)
pressure, u, v = predict(model, input_tensor, norm_stats, device=DEVICE)
speed = np.sqrt(u**2 + v**2)
cl = compute_cl(pressure, sdf, angle, velocity)
ood_score = distance_from_training(sdf)

if ood_score > 0.5:
    st.warning(
        "This geometry looks unlike the NACA airfoils the model was trained on. "
        "The prediction may be physically unreliable — the model extrapolates beyond "
        "what it has seen. (A deliberate demonstration of model limits.)"
    )

col_a, col_b, col_c = st.columns(3)
for col, label, value in [
    (col_a, "Lift coefficient C\u2097", f"{cl:.3f}"),
    (col_b, "Angle of attack", f"{angle}°"),
    (col_c, "Velocity", f"{velocity} m/s"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

def crop_to_view(field):
    x_min, x_max, y_min, y_max = DOMAIN
    vx_min, vx_max, vy_min, vy_max = VIEW
    h, w = field.shape
    i0 = int((vy_min - y_min) / (y_max - y_min) * h)
    i1 = int((vy_max - y_min) / (y_max - y_min) * h)
    j0 = int((vx_min - x_min) / (x_max - x_min) * w)
    j1 = int((vx_max - x_min) / (x_max - x_min) * w)
    return field[i0:i1, j0:j1]


def plot_field(field, title, cmap, label, profile_name, symmetric=False):
    field_c = crop_to_view(field)
    mask_c = crop_to_view(mask.astype(float))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    if symmetric:
        vmax = np.percentile(np.abs(field_c), 97)
        vmin = -vmax
    else:
        vmin, vmax = np.percentile(field_c, [2, 97])

    im = ax.imshow(field_c, cmap=cmap, origin="lower", extent=VIEW,
                   vmin=vmin, vmax=vmax, aspect="equal")
    gx = np.linspace(VIEW[0], VIEW[1], mask_c.shape[1])
    gy = np.linspace(VIEW[2], VIEW[3], mask_c.shape[0])
    ax.contour(gx, gy, mask_c, levels=[0.5], colors="white", linewidths=1.5)
    ax.contour(gx, gy, mask_c, levels=[0.5], colors="black", linewidths=0.6)

    ax.set_title(f"{title} — {profile_name}", fontsize=13, fontweight="bold", color="#888")
    ax.set_xlabel("x/c", color="#888")
    ax.set_ylabel("y/c", color="#888")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#888")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    cbar.ax.yaxis.label.set_color("#888")
    cbar.ax.tick_params(colors="#888")
    plt.tight_layout()
    return fig

tab1, tab2, tab3 = st.tabs(["Pressure", "Velocity", "Streamlines"])

with tab1:
    st.pyplot(plot_field(pressure, "Pressure field", "RdBu_r", "Pressure [Pa]", profile_name,
                         symmetric=True))
    st.caption("Blue = suction (low pressure) above the airfoil — this is what generates "
               "lift. Red = higher pressure below and at the stagnation point.")

with tab2:
    st.pyplot(plot_field(speed, "Velocity magnitude", "viridis", "|U| [m/s]", profile_name))
    st.caption("Flow accelerates over the upper surface (higher speed → lower pressure). Darker region behind is the wake.")

with tab3:
    u_c = crop_to_view(u)
    v_c = crop_to_view(v)
    speed_c = crop_to_view(speed)
    mask_c = crop_to_view(mask.astype(float))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ny, nx = u_c.shape
    Y, X = np.mgrid[VIEW[2]:VIEW[3]:complex(ny), VIEW[0]:VIEW[1]:complex(nx)]
    strm = ax.streamplot(X, Y, u_c, v_c, density=2.2, color=speed_c,
                         cmap="viridis", linewidth=0.9, arrowsize=0.8)
    gx = np.linspace(VIEW[0], VIEW[1], nx)
    gy = np.linspace(VIEW[2], VIEW[3], ny)
    ax.contourf(gx, gy, mask_c, levels=[0.5, 1.5], colors="#1a2b4a")
    ax.contour(gx, gy, mask_c, levels=[0.5], colors="white", linewidths=1.2)
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect("equal")
    ax.set_title(f"Streamlines — {profile_name}", fontsize=13, fontweight="bold", color="#888")
    ax.set_xlabel("x/c", color="#888")
    ax.set_ylabel("y/c", color="#888")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#888")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Streamlines trace the flow path. Color indicates local speed.")

if aoa_sweep:
    st.markdown("### Lift coefficient vs. angle of attack")
    cl_list = []
    for aoa in range(-5, 16, 1):
        if input_mode == "Draw shape":
            sdf_sweep, mask_sweep = canvas_to_sdf(canvas_result.json_data, canvas_size=400)
        elif input_mode == "NACA generator":
            sdf_sweep, mask_sweep = naca_to_sdf(m / 100, p / 10, t / 100, angle_deg=aoa)
        elif uploaded_coords is not None:
            sdf_sweep, mask_sweep = coordinates_to_sdf(uploaded_coords[0], uploaded_coords[1], angle_deg=aoa)
        input_tensor = build_input(sdf_sweep, aoa, velocity, norm_stats)
        pressure, u, v = predict(model, input_tensor, norm_stats, device=DEVICE)
        cl_list.append(compute_cl(pressure, sdf_sweep, aoa, velocity))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.plot(range(-5, 16, 1), cl_list, marker='o', color="#2d6cdf")
    ax.set_xlabel("Angle of attack [°]", color="#888")
    ax.set_ylabel("C\u2097", color="#888")
    ax.set_title(f"AoA Sweep — {profile_name}",  fontsize=13, fontweight="bold", color="#888")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#888")
    ax.set_xticks(range(-5, 16, 5))
    st.pyplot(fig)

st.markdown("---")
st.markdown(
    "<p style='opacity:0.55; font-size:0.85rem'>"
    "U-Net trained on the AirfRANS dataset. Predictions are "
    "approximate, intended for demonstration and rapid design exploration, not for "
    "engineering decisions. Drag (C\u1d05) is not predicted as it requires accurate "
    "near-wall shear stress that the 128×128 grid does not resolve."
    "</p>",
    unsafe_allow_html=True,
)
