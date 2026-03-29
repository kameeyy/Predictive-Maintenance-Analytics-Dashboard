import streamlit as st
from dashboard_utils import load_css

load_css()


def show():
    st.title("ℹ️ About")
    st.divider()

    st.markdown("""
    ### FYP: Predictive Maintenance for Centrifugal Pump Bearings
    **Using Health Index, Deep Learning & Explainable AI**
    """)

    # ── Dataset ────────────────────────────────────────────────────────────
    st.subheader("📦 Dataset")
    st.markdown("""
    **XJTU-SY Bearing Dataset** — Xi'an Jiaotong University, 15 bearings, 3 operating conditions.

    | Property | Detail |
    |---|---|
    | Bearings | 15 total — 5 per condition |
    | Conditions | 35Hz12kN · 37.5Hz11kN · 40Hz10kN |
    | Sensors | 2 accelerometers — Horizontal (H) & Vertical (V) |
    | Sampling rate | 25,600 Hz |
    | Data format | One CSV per minute (run-to-failure) |
    | Features extracted | 34 per sample (16 time-domain × 2ch + 4 freq × 2ch + 2 cross-channel) |

    > The dataset is collected from a rolling element bearing testrig.
    > It is applicable to centrifugal pump health monitoring because centrifugal pumps
    > use rolling element bearings as their primary rotating component.
    """)

    st.divider()

    # ── Train/test split ───────────────────────────────────────────────────
    st.subheader("✂️ Train / Test Split Strategy")
    st.markdown("""
    **Leave-One-Bearing-Out** — not a random 80/20 split.

    | Split | Bearings |
    |---|---|
    | Train | Bearings 1, 2, 3, 4 (per condition — complete run-to-failure trajectories) |
    | Test  | Bearing 5 (per condition — completely unseen, never seen during training) |

    This is stricter than 80/20 because it tests genuine cross-bearing generalisation —
    the model must predict health on a bearing it has never encountered.
    A random 80/20 split would leak correlated samples from the same bearing into both sets.
    """)

    st.divider()

    # ── Models ─────────────────────────────────────────────────────────────
    st.subheader("🤖 Models (v9)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🌲 Random Forest**
        - 500 trees, max_depth=10
        - Input: flattened 20-step window (680 features)
        - Non-sequential baseline
        - Best for cross-condition stability
        - Most consistent: Mean R²=**+0.370**
        """)
    with col2:
        st.markdown("""
        **🔁 BiLSTM + Attention**
        - BiLSTM(128) → BiLSTM(64) → Multi-Head Attention (4 heads)
        - Residual + LayerNorm → GlobalAvgPool → Dense
        - 375,233 parameters
        - Best single result: R²=**0.9754** (35Hz12kN)
        - Huber loss, WINDOW=20
        """)
    with col3:
        st.markdown("""
        **🔀 CNN-BiLSTM + Attention**
        - Conv1D(64) → Conv1D(128) → BiLSTM(64) → Multi-Head Attention
        - Residual + LayerNorm → GlobalAvgPool → Dense
        - 174,849 parameters
        - R²=**0.9735** (35Hz12kN)
        - CNN extracts local transient patterns before LSTM
        """)

    st.divider()

    # ── v9 Results ─────────────────────────────────────────────────────────
    st.subheader("📊 v9 Results Summary")
    st.markdown("""
    | Model | 35Hz12kN | 37.5Hz11kN | 40Hz10kN | Mean R² |
    |---|---|---|---|---|
    | Random Forest | 0.7137 ✅ | 0.6082 👍 | -0.2133 ⚠️ | **+0.370** |
    | BiLSTM+Attn | **0.9754 ✅** | 0.5172 👍 | -1.5014 ⚠️ | -0.003 |
    | CNN-BiLSTM+Attn | 0.9735 ✅ | -0.1206 ⚠️ | -1.0046 ⚠️ | -0.051 |

    **40Hz10kN note:** Training bearings average 1,730 samples; test bearing only 114 samples (22× mismatch).
    All models fail on this condition due to structural dataset limitation, not model design failure.
    This is consistent with findings in the XJTU-SY literature.
    """)

    st.divider()

    # ── Failure modes ──────────────────────────────────────────────────────
    st.subheader("⚠️ Bearing Failure Modes")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("""
        **🔴 Fatigue Spalling**

        Surface material breaks away from bearing raceway under cyclic loading.

        *Vibration signature:*
        - High kurtosis (>3)
        - High crest factor
        - Impulsive spikes at bearing defect frequencies
        """)
    with c2:
        st.warning("""
        **🟡 Abrasive Wear**

        Gradual surface material removal due to hard particle contamination.

        *Vibration signature:*
        - Rising RMS and energy
        - Broadband spectral energy increase
        - Gradual signal amplitude growth
        """)
    with c3:
        st.info("""
        **🔵 Lubrication Failure**

        Insufficient or degraded lubricant causing metal-to-metal contact.

        *Vibration signature:*
        - Rising spectral entropy
        - Elevated crest factor
        - High-frequency content increase
        """)

    st.divider()

    # ── Dashboard note ─────────────────────────────────────────────────────
    st.subheader("🖥️ Dashboard Mode")
    st.warning("""
    **Batch Prediction Mode** — This dashboard visualises pre-exported CSV predictions
    generated by the trained models (not live inference).

    For live deployment: export `.keras` models to a FastAPI or MQTT streaming pipeline
    that feeds real-time sensor data → runs the scaler + sliding window → calls `model.predict()`.
    """)

    st.divider()
    st.markdown("""
    **Stack:** Python · TensorFlow 2.19 · scikit-learn · SHAP · Streamlit · Matplotlib · NumPy · Pandas
    """)
