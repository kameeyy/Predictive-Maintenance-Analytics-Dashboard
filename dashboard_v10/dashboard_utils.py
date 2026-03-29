import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore')

BASE = os.path.dirname(__file__)

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, 'data', name))

@st.cache_data
def load_all_data():
    try:
        # Try v9 filenames first, fall back to original names
        try:
            df_results = load_csv('model_results_v9.csv')
        except Exception:
            df_results = load_csv('model_results.csv')
        try:
            df_preds = load_csv('predictions_v9.csv')
        except Exception:
            df_preds = load_csv('predictions.csv')
        df_shap       = load_csv('shap_values_rf.csv')
        df_feats      = load_csv('test_bearing_features.csv')
        try:
            df_shap_bilstm = load_csv('shap_values_bilstm.csv')
        except Exception:
            df_shap_bilstm = None
        return df_results, df_preds, df_shap, df_feats, True, df_shap_bilstm
    except Exception:
        return None, None, None, None, False, None

CONDITIONS  = ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']
MODEL_NAMES = ['Random Forest', 'BiLSTM+Attn', 'CNN-BiLSTM+Attn']
COLORS      = {
    'Random Forest':   '#2196F3',
    'BiLSTM+Attn':     '#4CAF50',
    'CNN-BiLSTM+Attn': '#E91E63',
}

# ── v9 Final Results ───────────────────────────────────────────────────────
RESULTS = {
    '35Hz12kN': {
        'Random Forest':   {'R2': 0.7137, 'RMSE': 0.1230, 'MAE': 0.1092},
        'BiLSTM+Attn':     {'R2': 0.9754, 'RMSE': 0.0361, 'MAE': 0.0293},
        'CNN-BiLSTM+Attn': {'R2': 0.9735, 'RMSE': 0.0374, 'MAE': 0.0255},
    },
    '37.5Hz11kN': {
        'Random Forest':   {'R2': 0.6082, 'RMSE': 0.1694, 'MAE': 0.1546},
        'BiLSTM+Attn':     {'R2': 0.5172, 'RMSE': 0.1881, 'MAE': 0.1799},
        'CNN-BiLSTM+Attn': {'R2':-0.1206, 'RMSE': 0.2865, 'MAE': 0.2728},
    },
    '40Hz10kN': {
        'Random Forest':   {'R2':-0.2133, 'RMSE': 0.2140, 'MAE': 0.1879},
        'BiLSTM+Attn':     {'R2':-1.5014, 'RMSE': 0.3073, 'MAE': 0.2823},
        'CNN-BiLSTM+Attn': {'R2':-1.0046, 'RMSE': 0.2751, 'MAE': 0.2573},
    },
}


def draw_pump_schematic(hi_value: float = 1.0, condition: str = '35Hz12kN') -> plt.Figure:
    """Draw a centrifugal pump schematic that accurately reflects the XJTU-SY dataset.
    ONE test bearing with TWO sensors (Horizontal + Vertical) mounted on it.
    Bearing colour changes with HI: green=healthy, orange=degrading, red=critical.
    """
    if hi_value > 0.6:
        bearing_color  = '#4CAF50'
        bearing_status = 'Healthy'
    elif hi_value > 0.3:
        bearing_color  = '#FF9800'
        bearing_status = 'Degrading'
    else:
        bearing_color  = '#f44336'
        bearing_status = 'Critical'

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(5, 6.75, 'Centrifugal Pump — Cross Section Schematic',
            ha='center', va='center', fontsize=10,
            color='white', fontweight='bold', zorder=20)
    ax.text(0.2, 6.75, f'Condition: {condition}',
            fontsize=8, color='#90A4AE', zorder=20)

    # ── Volute casing ──────────────────────────────────────────────────────
    volute       = plt.Circle((5, 3.6), 2.4, color='#37474F', zorder=2)
    volute_inner = plt.Circle((5, 3.6), 2.2, color='#263238', zorder=3)
    ax.add_patch(volute)
    ax.add_patch(volute_inner)

    # Spiral outline
    theta = np.linspace(0, 1.8 * np.pi, 200)
    r  = 2.0 + 0.18 * (theta / (1.8 * np.pi))
    ax.plot(5 + r * np.cos(theta), 3.6 + r * np.sin(theta),
            color='#546E7A', lw=3, zorder=4)

    # ── Impeller ───────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((5, 3.6), 1.5, color='#1565C0', zorder=5, alpha=0.9))
    ax.add_patch(plt.Circle((5, 3.6), 0.5, color='#0D47A1', zorder=6))
    for angle in np.linspace(0, 2 * np.pi, 7)[:-1]:
        ax.plot([5 + 0.5 * np.cos(angle), 5 + 1.4 * np.cos(angle + 0.35)],
                [3.6 + 0.5 * np.sin(angle), 3.6 + 1.4 * np.sin(angle + 0.35)],
                color='#42A5F5', lw=2.5, zorder=7)

    # ── Shaft ──────────────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle((0.3, 3.35), 9.4, 0.5,  color='#78909C', zorder=8))
    ax.add_patch(plt.Rectangle((0.3, 3.60), 9.4, 0.15, color='#B0BEC5', alpha=0.4, zorder=9))

    # ── Single test bearing (drive end — left side, as in XJTU-SY) ────────
    bx = 1.5
    by = 3.6

    # Glow effect if critical
    if hi_value <= 0.3:
        ax.add_patch(plt.Circle((bx, by), 0.95, color='#f44336', alpha=0.15, zorder=9))
        ax.add_patch(plt.Circle((bx, by), 0.82, color='#f44336', alpha=0.12, zorder=9))

    # Outer ring
    ax.add_patch(plt.Circle((bx, by), 0.75, color='#455A64', zorder=10))
    # Inner race (health colour)
    ax.add_patch(plt.Circle((bx, by), 0.48, color=bearing_color, zorder=11, alpha=0.9))
    # Rolling elements
    for ba in np.linspace(0, 2 * np.pi, 10)[:-1]:
        ax.add_patch(plt.Circle(
            (bx + 0.615 * np.cos(ba), by + 0.615 * np.sin(ba)),
            0.09, color='#CFD8DC', zorder=12))
    # Shaft through bearing
    ax.add_patch(plt.Rectangle((bx - 0.48, 3.35), 0.96, 0.5, color='#78909C', zorder=13))

    # Bearing label
    ax.text(bx, by - 1.0, 'Test Bearing\n(XJTU-SY)', ha='center', va='center',
            fontsize=7, color=bearing_color, fontweight='bold', zorder=20)

    # ── Right side — plain shaft support (no monitored bearing) ───────────
    bx2 = 8.5
    ax.add_patch(plt.Circle((bx2, by), 0.55, color='#37474F', zorder=10))
    ax.add_patch(plt.Circle((bx2, by), 0.35, color='#455A64', zorder=11))
    ax.add_patch(plt.Rectangle((bx2 - 0.35, 3.35), 0.70, 0.5, color='#78909C', zorder=13))
    ax.text(bx2, by - 0.85, 'Shaft Support', ha='center', va='center',
            fontsize=6.5, color='#546E7A', zorder=20)

    # ── TWO sensors on the SAME test bearing (H and V) ────────────────────
    # Sensor H — horizontal (top of bearing)
    sh_x, sh_y = bx, by + 0.95
    ax.add_patch(plt.Circle((sh_x, sh_y + 0.35), 0.22, color='#FFC107', zorder=15, alpha=0.9))
    ax.text(sh_x, sh_y + 0.35, '~', ha='center', va='center',
            fontsize=9, color='#0e1117', fontweight='bold', zorder=16)
    ax.plot([sh_x, sh_x], [by + 0.75, sh_y + 0.13],
            color='#FFC107', lw=1.2, ls='--', alpha=0.7, zorder=14)
    ax.text(sh_x, sh_y + 0.85, 'Sensor (H)\nHorizontal', ha='center', va='bottom',
            fontsize=6.5, color='#FFC107', zorder=16)

    # Sensor V — vertical (side of bearing)
    sv_x, sv_y = bx + 0.95, by
    ax.add_patch(plt.Circle((sv_x + 0.35, sv_y), 0.22, color='#FF7043', zorder=15, alpha=0.9))
    ax.text(sv_x + 0.35, sv_y, '~', ha='center', va='center',
            fontsize=9, color='#0e1117', fontweight='bold', zorder=16)
    ax.plot([bx + 0.75, sv_x + 0.13], [sv_y, sv_y],
            color='#FF7043', lw=1.2, ls='--', alpha=0.7, zorder=14)
    ax.text(sv_x + 0.35, sv_y + 0.45, 'Sensor (V)\nVertical', ha='center', va='bottom',
            fontsize=6.5, color='#FF7043', zorder=16)

    # ── Sensor signal lines to data label ─────────────────────────────────
    ax.annotate('', xy=(3.2, 5.5), xytext=(sh_x + 0.22, sh_y + 0.35),
                arrowprops=dict(arrowstyle='->', color='#FFC107', lw=1.2, alpha=0.6))
    ax.annotate('', xy=(3.2, 5.5), xytext=(sv_x + 0.57, sv_y),
                arrowprops=dict(arrowstyle='->', color='#FF7043', lw=1.2, alpha=0.6))

    data_box = FancyBboxPatch((3.2, 5.1), 2.2, 0.8,
                               boxstyle='round,pad=0.08',
                               facecolor='#1e2a3a', edgecolor='#4FC3F7',
                               linewidth=1.5, zorder=18)
    ax.add_patch(data_box)
    ax.text(4.3, 5.5, '34 Features Extracted\n(17 H-channel + 17 V-channel)',
            ha='center', va='center', fontsize=6.5, color='#4FC3F7', zorder=19)

    # ── Pipes ──────────────────────────────────────────────────────────────
    # Discharge (top)
    ax.add_patch(plt.Rectangle((4.4, 5.8), 1.2, 1.0, color='#37474F', zorder=4))
    ax.add_patch(plt.Rectangle((4.55, 5.8), 0.9, 1.0, color='#1a2332', zorder=5))
    ax.annotate('', xy=(5, 6.85), xytext=(5, 5.95),
                arrowprops=dict(arrowstyle='->', color='#4FC3F7', lw=2))

    # Suction (left)
    ax.add_patch(plt.Rectangle((0, 3.25), 2.05, 0.7, color='#37474F', zorder=4))
    ax.add_patch(plt.Rectangle((0, 3.40), 2.05, 0.4, color='#1a2332', zorder=5))
    ax.annotate('', xy=(1.9, 3.6), xytext=(0.2, 3.6),
                arrowprops=dict(arrowstyle='->', color='#4FC3F7', lw=2))

    # ── Static labels ──────────────────────────────────────────────────────
    for lx, ly, txt, col, fs, fw in [
        (5,    6.3,  'Discharge',     '#4FC3F7', 8,   'bold'),
        (5,    1.2,  'Volute Casing', '#90A4AE', 8,   'normal'),
        (5,    3.6,  'Impeller',      '#ffffff', 7.5, 'bold'),
        (0.15, 3.6,  'Suction',       '#4FC3F7', 7.5, 'bold'),
    ]:
        ax.text(lx, ly, txt, ha='center', va='center',
                fontsize=fs, color=col, fontweight=fw, zorder=20)

    # ── HI status box ──────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((6.8, 0.15), 2.9, 1.5,
                                boxstyle='round,pad=0.1',
                                facecolor='#1e1e2e', edgecolor=bearing_color,
                                linewidth=2, zorder=20))
    ax.text(8.25, 1.35, 'Bearing HI',      ha='center', fontsize=7.5, color='#aaa',          zorder=21)
    ax.text(8.25, 0.88, f'{hi_value:.3f}', ha='center', fontsize=16,  color=bearing_color,
            fontweight='bold', zorder=21)
    ax.text(8.25, 0.45, bearing_status,    ha='center', fontsize=8,   color=bearing_color,   zorder=21)

    # ── HI progress bar ────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle((0.3, 0.20), 6.0,            0.38, color='#263238',     zorder=20))
    ax.add_patch(plt.Rectangle((0.3, 0.20), 6.0 * hi_value, 0.38, color=bearing_color, zorder=21, alpha=0.85))
    ax.text(0.3,  0.70, 'HI', fontsize=7,   color='#aaa', zorder=22)
    ax.text(0.3,  0.05, '0',  fontsize=6.5, color='#aaa', zorder=22)
    ax.text(6.15, 0.05, '1',  fontsize=6.5, color='#aaa', zorder=22)

    for thresh, col in [(0.3, '#FF9800'), (0.1, '#f44336')]:
        tx = 0.3 + 6.0 * thresh
        ax.plot([tx, tx], [0.17, 0.62], color=col, lw=1.2, ls='--', zorder=22, alpha=0.8)

    plt.tight_layout(pad=0.3)
    return fig


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
