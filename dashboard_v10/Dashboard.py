import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bearing Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from dashboard_utils import load_css
from streamlit_option_menu import option_menu
from pages import Model_Results, Predictions, Health_Index, SHAP

load_css()

# Fix 1: default_index=0 ensures Dashboard tab is selected first, not SHAP
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Predictions", "Health Index", "SHAP"],
    icons=["speedometer", "activity", "heart-pulse", "brain"],
    orientation="horizontal",
    default_index=0,
    styles={
        "nav-link-selected": {"background-color": "#00C9B1"},
    }
)

if selected == "Dashboard":
    Model_Results.show()
elif selected == "Predictions":
    Predictions.show()
elif selected == "Health Index":
    Health_Index.show()
elif selected == "SHAP":
    SHAP.show()
