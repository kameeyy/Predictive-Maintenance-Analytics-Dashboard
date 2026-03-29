import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dashboard_utils import load_all_data, load_css


load_css()

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_all_bearings_hi():
    for fname in ('all_bearings_hi_v9.csv', 'all_bearings_hi.csv'):
        path = os.path.join(BASE, 'data', fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

df_all_hi     = load_all_bearings_hi()
ALL_HI_LOADED = df_all_hi is not None

CONDITIONS   = ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']
TRAIN_COLORS = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC']
TEST_COLOR   = '#f44336'


def show():
    df_results, df_preds, df_shap, df_feats, DATA_LOADED, _ = load_all_data()
    st.title("📈 Health Index")
    st.divider()

    st.markdown("""
    ### Formula
    > **HI = 1 − (combined_RMS − RMS_min) / (RMS_max − RMS_min)**
    >
    > combined_RMS = √((h_rms² + v_rms²) / 2)   — combines both H and V sensor axes

    | HI Range | Status |
    |---|---|
    | 0.7 – 1.0 | ✅ Healthy — normal operation |
    | 0.3 – 0.7 | ⚠️ Degrading — monitor closely |
    | 0.1 – 0.3 | 🔶 Warning — schedule maintenance |
    | 0.0 – 0.1 | 🔴 Critical — imminent failure |

    Smoothed with a **5-point centred rolling average** to reduce measurement noise.
    """)

    st.divider()

    # ── Per-condition HI curves ─────────────────────────────────────────────
    st.subheader("Health Index — All Bearings by Condition")
    cond_sel = st.selectbox("Select Condition", CONDITIONS)

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='#0e1117')
    ax.set_facecolor('#1e1e2e')

    if ALL_HI_LOADED:
        sub = df_all_hi[df_all_hi['Condition'] == cond_sel]
        train_idx = 0
        for bname in sorted(sub['Bearing'].unique()):
            b_data  = sub[sub['Bearing'] == bname].sort_values('Sample')
            is_test = bool(b_data['IsTest'].iloc[0])
            hi = b_data['HI'].values
            t  = b_data['Sample'].values
            if is_test:
                ax.plot(t, hi, color=TEST_COLOR, lw=2.5, ls='--',
                        label=f'{bname} (TEST)', zorder=5)
            else:
                col = TRAIN_COLORS[train_idx % len(TRAIN_COLORS)]
                ax.plot(t, hi, color=col, lw=1.5, alpha=0.85, label=bname)
                train_idx += 1
    else:
        st.warning("all_bearings_hi_v9.csv not found — upload from Google Drive FYP_Dashboard/data/")

    ax.axhline(0.3, color='#FF9800', lw=1.2, ls=':', alpha=0.9, label='Warning (0.3)')
    ax.axhline(0.1, color='#f44336', lw=1.2, ls=':', alpha=0.9, label='Critical (0.1)')
    ax.set_xlabel('Sample (minutes)', color='white')
    ax.set_ylabel('Health Index',     color='white')
    ax.set_title(f'All Bearings HI — {cond_sel}', color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.legend(fontsize=8.5, facecolor='#1e1e2e', labelcolor='white',
              ncol=3, loc='lower left')
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Summary table ──────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Bearing Summary — {cond_sel}")
    if ALL_HI_LOADED:
        sub  = df_all_hi[df_all_hi['Condition'] == cond_sel]
        rows = []
        for bname in sorted(sub['Bearing'].unique()):
            b_data  = sub[sub['Bearing'] == bname]
            is_test = bool(b_data['IsTest'].iloc[0])
            hi_vals = b_data['HI'].values
            rows.append({
                'Bearing':  bname,
                'Role':     '🔴 Test' if is_test else '🔵 Train',
                'Samples':  len(hi_vals),
                'HI Start': f'{hi_vals[0]:.3f}',
                'HI End':   f'{hi_vals[-1]:.3f}',
                'Min HI':   f'{hi_vals.min():.3f}',
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── All 3 conditions overview ──────────────────────────────────────────
    st.divider()
    st.subheader("All Conditions Overview")
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor='#0e1117')
    fig2.suptitle('Health Index — All Bearings, All Conditions',
                  color='white', fontsize=11)
    for ax, cond in zip(axes, CONDITIONS):
        ax.set_facecolor('#1e1e2e')
        if ALL_HI_LOADED:
            sub = df_all_hi[df_all_hi['Condition'] == cond]
            train_idx = 0
            for bname in sorted(sub['Bearing'].unique()):
                b_data  = sub[sub['Bearing'] == bname].sort_values('Sample')
                is_test = bool(b_data['IsTest'].iloc[0])
                hi = b_data['HI'].values
                t  = b_data['Sample'].values
                if is_test:
                    ax.plot(t, hi, color=TEST_COLOR, lw=2, ls='--', label='Test', zorder=5)
                else:
                    col = TRAIN_COLORS[train_idx % len(TRAIN_COLORS)]
                    ax.plot(t, hi, color=col, lw=1.2, alpha=0.75)
                    train_idx += 1
        ax.axhline(0.3, color='#FF9800', lw=1,   ls=':', alpha=0.7)
        ax.axhline(0.1, color='#f44336', lw=1,   ls=':', alpha=0.7)
        ax.set_title(cond, color='white', fontsize=9)
        ax.set_xlabel('Sample',   color='white', fontsize=8)
        ax.set_ylabel('HI',       color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        ax.spines[:].set_color('#333')
        ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Feature list ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("34 Features Extracted per Sample")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**⏱ Time Domain (12 × 2 channels = 24)**")
        st.markdown("RMS, Peak, Peak-to-Peak, Kurtosis, Skewness, Std Dev, "
                    "Energy, Log Energy, Mean Absolute, Crest Factor, Shape Factor, IQR")
    with c2:
        st.markdown("**🔊 Frequency Domain (4 × 2 channels = 8)**")
        st.markdown("Frequency Centroid, Spectral Entropy, Peak Frequency 1, Peak Amplitude 1")
    with c3:
        st.markdown("**🔗 Cross-Channel (2)**")
        st.markdown("H-V Correlation, H-V RMS Ratio")
