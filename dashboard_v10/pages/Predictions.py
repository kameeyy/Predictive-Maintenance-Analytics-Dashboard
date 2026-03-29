import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from dashboard_utils import (load_css, COLORS, CONDITIONS, MODEL_NAMES,
                              RESULTS, draw_pump_schematic, load_all_data)

load_css()


def show():
    df_results, df_preds, df_shap, df_feats, DATA_LOADED, _ = load_all_data()

    st.title("⚙️ Live Bearing Health Monitor")
    st.markdown(
        "Select an operating condition and model below. "
        "Use the **time step slider** to simulate monitoring the bearing minute by minute "
        "and watch the pump respond in real time."
    )
    st.divider()

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        cond_sel = st.selectbox("⚡ Operating Condition", CONDITIONS)
    with col_ctrl2:
        model_sel = st.selectbox("🤖 Prediction Model", MODEL_NAMES)
    with col_ctrl3:
        if DATA_LOADED and df_preds is not None:
            sub_all    = df_preds[(df_preds['Condition'] == cond_sel) &
                                   (df_preds['Model']     == model_sel)]
            max_sample = max(len(sub_all) - 1, 1)
        else:
            max_sample = 50
        sample_idx = st.slider("🕐 Time Step (minutes)", 0, max_sample,
                               min(max_sample, 20))

    st.divider()

    r2   = RESULTS[cond_sel][model_sel]['R2']
    rmse = RESULTS[cond_sel][model_sel]['RMSE']
    mae  = RESULTS[cond_sel][model_sel]['MAE']

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²",   f"{r2:+.4f}",
              "✅ Excellent" if r2 > 0.7 else "👍 Acceptable" if r2 > 0.4 else "⚠️ Poor")
    m2.metric("RMSE", f"{rmse:.4f}")
    m3.metric("MAE",  f"{mae:.4f}")

    # ── Load actual / predicted arrays ────────────────────────────────────
    if DATA_LOADED and df_preds is not None:
        sub = df_preds[(df_preds['Condition'] == cond_sel) &
                       (df_preds['Model']     == model_sel)].reset_index(drop=True)
        if len(sub) == 0:
            st.warning(f"No prediction data found for {model_sel} | {cond_sel}.")
            return
        step_col = 'Step' if 'Step' in sub.columns else 'Sample'
        sub = sub.sort_values(step_col).reset_index(drop=True)
        yt  = sub['Actual'].values
        yp  = sub['Predicted'].values
    else:
        st.warning("Prediction data not loaded. Using placeholder data.")
        np.random.seed(42)
        n   = 50
        t   = np.linspace(0, 1, n)
        yt  = np.clip(1 - (t > 0.5) * (t - 0.5) / 0.5 + 0.01 * np.random.randn(n), 0, 1)
        yp  = np.clip(yt + np.random.normal(0, rmse, n), 0, 1)

    sample_idx  = min(sample_idx, len(yt) - 1)
    current_hi  = float(yt[sample_idx])
    pred_hi     = float(yp[sample_idx])
    status_str  = ("✅ Healthy"   if current_hi > 0.6 else
                   "⚠️ Degrading" if current_hi > 0.3 else
                   "🔴 Critical")
    m4.metric("Current HI", f"{current_hi:.3f}", status_str)

    st.divider()

    col_pump, col_chart = st.columns([1, 1])

    with col_pump:
        st.subheader("🔧 Pump Equipment Status")
        fig_pump = draw_pump_schematic(hi_value=current_hi, condition=cond_sel)
        st.pyplot(fig_pump)
        plt.close()
        st.caption(
            f"Minute **{sample_idx}** — "
            f"Actual HI: **{current_hi:.3f}** | "
            f"Predicted: **{pred_hi:.3f}** — {status_str}"
        )

    with col_chart:
        st.subheader("📈 Health Index Over Time")
        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 7), facecolor='#0e1117')

        # ── Time-series ───────────────────────────────────────────────────
        ax = axes2[0]
        ax.set_facecolor('#1e1e2e')
        t_arr = np.arange(len(yt))
        ax.fill_between(t_arr, yt, alpha=0.08, color='white')
        ax.plot(t_arr, yt, 'w-',  lw=2, label='Actual HI')
        ax.plot(t_arr, yp, '--',  lw=2, color=COLORS[model_sel], label='Predicted HI')
        ax.fill_between(t_arr, yt, yp, alpha=0.10, color=COLORS[model_sel])
        ax.axhline(0.3, color='#FF9800', lw=1.2, ls=':', alpha=0.9, label='⚠️ Warning (0.3)')
        ax.axhline(0.1, color='#f44336', lw=1.2, ls=':', alpha=0.9, label='🔴 Critical (0.1)')
        ax.axvline(sample_idx, color='#FFC107', lw=1.5, ls='--', alpha=0.8, label='Now')
        ax.scatter([sample_idx], [current_hi], color='#FFC107', s=80, zorder=10)
        ax.set_xlabel('Time (minutes)', color='white')
        ax.set_ylabel('Health Index',   color='white')
        ax.set_title(f'{model_sel} | {cond_sel}', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        ax.legend(fontsize=7.5, facecolor='#1e1e2e', labelcolor='white', ncol=2)
        ax.set_ylim(-0.05, 1.1)

        # ── Scatter ───────────────────────────────────────────────────────
        ax2 = axes2[1]
        ax2.set_facecolor('#1e1e2e')
        ax2.scatter(yt, yp, alpha=0.6, s=25, c=COLORS[model_sel],
                    edgecolors='white', linewidths=0.2)
        ax2.plot([0, 1], [0, 1], 'w--', lw=1.5, alpha=0.5, label='Perfect prediction')
        ax2.scatter([current_hi], [pred_hi], color='#FFC107', s=120, zorder=10,
                    label=f'Now (min {sample_idx})')
        ax2.set_xlabel('Actual HI',    color='white')
        ax2.set_ylabel('Predicted HI', color='white')
        ax2.set_title(f'Actual vs Predicted  R²={r2:.3f}', color='white', fontsize=10)
        ax2.tick_params(colors='white')
        ax2.spines[:].set_color('#333')
        ax2.legend(fontsize=7.5, facecolor='#1e1e2e', labelcolor='white')
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.divider()

    # ── Maintenance alert ─────────────────────────────────────────────────
    if current_hi > 0.6:
        st.success(
            f"✅ **Bearing is Healthy** at minute {sample_idx}. "
            f"Health Index: **{current_hi:.3f}**. Continue normal operation — no action required."
        )
    elif current_hi > 0.3:
        st.warning(
            f"⚠️ **Early Degradation Detected** at minute {sample_idx}. "
            f"Health Index: **{current_hi:.3f}**. "
            f"Schedule an inspection within the next maintenance window."
        )
    else:
        st.error(
            f"🔴 **Critical Condition** at minute {sample_idx}. "
            f"Health Index: **{current_hi:.3f}**. "
            f"Bearing failure is imminent — immediate maintenance required!"
        )
