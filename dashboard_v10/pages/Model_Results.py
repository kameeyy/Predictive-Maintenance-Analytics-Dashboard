import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dashboard_utils import load_css, COLORS, CONDITIONS, MODEL_NAMES, RESULTS, load_all_data

load_css()


def show():
    df_results, df_preds, df_shap, df_feats, DATA_LOADED, _ = load_all_data()
    st.title("📊 Model Performance")
    st.divider()

    metric_choice = st.selectbox("Select Metric", ["R²", "RMSE", "MAE"])
    mk = {"R²": "R2", "RMSE": "RMSE", "MAE": "MAE"}[metric_choice]

    # ── Summary table — no Mean column ────────────────────────────────────
    rows = [{"Condition": c, **{m: RESULTS[c][m][mk] for m in MODEL_NAMES}}
            for c in CONDITIONS]
    df_tbl = pd.DataFrame(rows).set_index('Condition')
    st.dataframe(df_tbl.style.format('{:+.4f}'), use_container_width=True)

    st.divider()

    # ── Bar charts ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='#0e1117')
    for ax, cond in zip(axes, CONDITIONS):
        vals  = [RESULTS[cond][m][mk] for m in MODEL_NAMES]
        bars  = ax.bar(MODEL_NAMES, vals,
                       color=[COLORS[m] for m in MODEL_NAMES],
                       width=0.5, edgecolor='#333')
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='white', labelsize=7)
        ax.set_title(cond, color='white', fontsize=9)
        ax.spines[:].set_color('#333')
        if mk == 'R2':
            ax.axhline(0,   color='white',   lw=1, ls='--', alpha=0.4)
            ax.axhline(0.7, color='#4CAF50', lw=1, ls=':',  alpha=0.7, label='Good (0.7)')
            ax.axhline(0.5, color='#FF9800', lw=1, ls=':',  alpha=0.7, label='OK (0.5)')
            ax.set_ylim(-1.8, 1.2)
            ax.legend(fontsize=6.5, facecolor='#1e1e2e', labelcolor='white')
        else:
            ax.axhline(0.10, color='#4CAF50', lw=1, ls=':', alpha=0.7, label='Excellent')
            ax.axhline(0.15, color='#FF9800', lw=1, ls=':', alpha=0.7, label='Good')
            ax.set_ylim(0, 0.45)
            ax.legend(fontsize=6.5, facecolor='#1e1e2e', labelcolor='white')
        for bar, v in zip(bars, vals):
            ypos = v + (0.03 if v >= 0 else -0.12)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{v:+.3f}', ha='center', color='white',
                    fontsize=8, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Key findings ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("""
        **✅ 35Hz12kN — Deep Learning Wins**
        - BiLSTM+Attn: **R²=0.9754** ← best result
        - CNN-BiLSTM+Attn: **R²=0.9735**
        - Attention mechanism significantly improved both deep learning models
        - RMSE=0.036 — predictions within 3.6% of HI scale
        """)
    with col2:
        st.warning("""
        **👍 37.5Hz11kN — RF Most Stable**
        - Random Forest: **R²=0.608** — most reliable
        - BiLSTM+Attn: R²=0.517
        - CNN-BiLSTM+Attn: R²=-0.121
        - Inconsistent training set (42 vs 533 samples)
        - RF handles distribution inconsistency better
        """)
    with col3:
        st.error("""
        **⚠️ 40Hz10kN — Dataset Limitation**
        - All models negative R² — expected
        - Training bearings: 1,515–2,538 samples each
        - Test bearing: only **114 samples** (22× shorter)
        - Not a model failure — structural lifespan mismatch
        """)
