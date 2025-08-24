"""
MEDIS Pharmaceutical Sales Forecasting Dashboard

Main dashboard application that combines data analysis and ML forecasting capabilities.
This is a comprehensive, production-ready Streamlit application for pharmaceutical sales analytics.

Author: AI Assistant
Date: 2025-01-09
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import dashboard components
from components.data_analysis_tab import DataAnalysisTab
from components.ml_forecasting_tab import MLForecastingTab

# Page configuration
st.set_page_config(
    page_title="MEDIS Sales Forecasting Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/medis-forecasting',
        'Report a bug': 'https://github.com/your-repo/medis-forecasting/issues',
        'About': '''
        # MEDIS Pharmaceutical Sales Forecasting Dashboard

        **Version:** 2.0.0
        **Date:** January 2025

        A comprehensive machine learning platform for pharmaceutical sales forecasting
        with competitive intelligence and business analytics.

        Built with ❤️ for data-driven pharmaceutical sales planning.
        '''
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tab-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e74c3c;
    }
    .version-info {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: #666;
        font-size: 0.8rem;
        background: rgba(255,255,255,0.8);
        padding: 5px 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application"""

    # Main header
    st.markdown('<div class="main-header">💊 MEDIS Sales Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced ML-Powered Pharmaceutical Sales Analytics & Forecasting</div>', unsafe_allow_html=True)

    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🔮 ML Forecasting", "🔄 Automated Evaluation"])

    # Initialize session state for data persistence
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.analysis_results = None

    # Data Analysis Tab
    with tab1:
        st.markdown('<div class="tab-header">📊 Data Analysis & Business Intelligence</div>', unsafe_allow_html=True)

        # Initialize and render data analysis tab
        data_analysis_tab = DataAnalysisTab()

        # Add sidebar filters
        filters = data_analysis_tab.add_sidebar_filters()

        # Render the analysis
        data_analysis_tab.render()

        # Store results in session state for cross-tab access
        if data_analysis_tab.analysis_results:
            st.session_state.analysis_results = data_analysis_tab.analysis_results
            st.session_state.data_loaded = True

    # ML Forecasting Tab
    with tab2:
        st.markdown('<div class="tab-header">🔮 ML Forecasting & Model Comparison</div>', unsafe_allow_html=True)

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load and analyze data in the Data Analysis tab first.")
            st.info("📊 Go to the **Data Analysis** tab to load your pharmaceutical sales data and perform comprehensive analysis.")

            # Show placeholder content
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("🤖 **Model Training**\n\nTrain Prophet, XGBoost, and TimesFM models")

            with col2:
                st.info("📈 **Multi-Model Comparison**\n\nCompare ground truth vs predictions from all models on one chart")

            with col3:
                st.info("🔍 **Performance Metrics**\n\nMAPE, RMSE, R² scores for each model")

            st.markdown("---")
            st.markdown("**Available Models:**")
            st.markdown("• **Prophet** - Facebook's time series forecasting")
            st.markdown("• **XGBoost** - Gradient boosting with features")
            st.markdown("• **TimesFM** - Google's foundation model (coming soon)")
            st.markdown("• **Naive & Seasonal Naive** - Baseline methods")
            st.markdown("• **Moving Average** - Simple trend forecasting")
        else:
            st.success("✅ Data loaded successfully! ML forecasting is ready.")

            # Initialize and render ML forecasting tab
            ml_forecasting_tab = MLForecastingTab()
            ml_forecasting_tab.render()

    # Automated Evaluation Tab (Placeholder)
    with tab3:
        st.markdown('<div class="tab-header">🔄 Automated Model Evaluation</div>', unsafe_allow_html=True)

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load and analyze data in the Data Analysis tab first.")
            st.info("📊 Go to the **Data Analysis** tab to load your pharmaceutical sales data and perform comprehensive analysis.")
        else:
            st.info("🔄 Advanced evaluation features coming in Phase 3")
            st.markdown("**Future features:**")
            st.markdown("• Walk-forward validation across multiple time periods")
            st.markdown("• Statistical significance testing")
            st.markdown("• Model robustness analysis")
            st.markdown("• Scenario analysis capabilities")

    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-header">📋 Dashboard Info</div>', unsafe_allow_html=True)

    st.sidebar.info("""
    **MEDIS Sales Forecasting Dashboard**

    **Version:** 2.0.0
    **Data Range:** April 2018 - April 2025
    **Models:** Prophet, XGBoost, LSTM, Transformer, Ensemble
    **Focus:** ATOR (atorvastatin) cholesterol medication market
    """)

    # Quick actions
    st.sidebar.markdown("### 🚀 Quick Actions")

    if st.sidebar.button("🔄 Reload Data", help="Reload and re-analyze the pharmaceutical sales data"):
        st.session_state.data_loaded = False
        st.session_state.analysis_results = None
        st.rerun()

    if st.sidebar.button("📊 Generate Report", help="Generate comprehensive business analysis report"):
        if st.session_state.data_loaded:
            st.sidebar.success("📄 Report generation feature coming soon!")
        else:
            st.sidebar.error("Please load data first")

    if st.sidebar.button("🔮 Start Forecasting", help="Begin ML model training and forecasting"):
        if st.session_state.data_loaded:
            st.sidebar.success("🤖 ML forecasting feature coming soon!")
        else:
            st.sidebar.error("Please load data first")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
        'Built with ❤️ for pharmaceutical sales forecasting and competitive intelligence | '
        '<strong>MEDIS Sales Forecasting Dashboard v2.0.0</strong>'
        '</div>',
        unsafe_allow_html=True
    )

    # Version info
    st.markdown('<div class="version-info">v2.0.0 | January 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
