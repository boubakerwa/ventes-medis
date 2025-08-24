#!/usr/bin/env python3
"""
Streamlit Dashboard for Automated Model Evaluation

Interactive dashboard for comprehensive walk-forward validation
across multiple time periods and forecast horizons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from automated_model_evaluation import WalkForwardValidator
from forecasting_models import load_and_prepare_data

# Page config
st.set_page_config(
    page_title="MEDIS Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_evaluation_data():
    """Load and cache the evaluation data"""
    try:
        df = load_and_prepare_data('MEDIS_VENTES.xlsx')
        medis_df = df[df['laboratoire'] == 'MEDIS'].copy()
        competitive_data = df[df['laboratoire'] != 'MEDIS'].copy()
        
        # Create monthly time series
        monthly_sales = medis_df.groupby('date')['sales'].sum().fillna(0).sort_index()
        
        return monthly_sales, competitive_data, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def run_evaluation(monthly_sales, competitive_data, models_to_test, horizons, min_train_months):
    """Run the automated evaluation"""
    
    with st.spinner("Running comprehensive evaluation... This may take a few minutes."):
        validator = WalkForwardValidator(min_train_months=min_train_months, max_test_months=12)
        
        # Generate validation schedule first to show progress
        schedule = validator.generate_validation_schedule(monthly_sales, horizons)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_scenarios = len(schedule) * len(models_to_test)
        current_scenario = 0
        
        # Custom evaluation with progress tracking
        models = validator.initialize_models()
        models = {k: v for k, v in models.items() if k in models_to_test}
        
        all_results = []
        
        for i, config in enumerate(schedule):
            status_text.text(f"Scenario {i+1}/{len(schedule)}: {config['cutoff_date'].strftime('%Y-%m')} "
                           f"(Horizon: {config['forecast_horizon']}m)")
            
            # Prepare data for this scenario
            cutoff_idx = monthly_sales.index.get_loc(config['cutoff_date'])
            train_data = monthly_sales[:cutoff_idx + 1]
            test_data = monthly_sales[cutoff_idx + 1:cutoff_idx + 1 + config['forecast_horizon']]
            
            # Filter competitive data
            comp_data_filtered = None
            if competitive_data is not None:
                comp_data_filtered = competitive_data[
                    competitive_data['date'] <= config['cutoff_date']
                ].copy()
            
            # Test each model
            for model_name, model in models.items():
                current_scenario += 1
                progress = current_scenario / total_scenarios
                progress_bar.progress(progress)
                
                result = validator.train_and_evaluate_model(
                    model_name, model, train_data, test_data, comp_data_filtered, config
                )
                all_results.append(result)
        
        status_text.text("Evaluation complete!")
        progress_bar.progress(1.0)
        
        results_df = pd.DataFrame(all_results)
        validator.results = results_df
        
        return validator, results_df

def create_interactive_heatmap(results_df, metric='MAPE'):
    """Create interactive heatmap with Plotly"""
    
    # Prepare data for heatmap
    heatmap_data = results_df.pivot_table(
        values=metric,
        index='cutoff_date',
        columns=['model', 'forecast_horizon'],
        aggfunc='mean'
    )
    
    # Create multi-level column names for better display
    column_names = []
    for col in heatmap_data.columns:
        model, horizon = col
        column_names.append(f"{model}<br>{horizon}m")
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=column_names,
        y=[d.strftime('%Y-%m') for d in heatmap_data.index],
        colorscale='RdYlGn_r' if metric in ['MAPE', 'RMSE'] else 'RdYlGn',
        text=np.round(heatmap_data.values, 1),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title=f'{metric} (%)')
    ))
    
    fig.update_layout(
        title=f'Model Performance Heatmap - {metric}<br><sub>{"Lower is Better" if metric in ["MAPE", "RMSE"] else "Higher is Better"}</sub>',
        xaxis_title='Model & Forecast Horizon',
        yaxis_title='Cutoff Date',
        height=600
    )
    
    return fig

def create_temporal_performance_chart(results_df, metric='MAPE'):
    """Create temporal performance line chart"""
    
    fig = go.Figure()
    
    models = results_df['model'].unique()
    colors = px.colors.qualitative.Set1[:len(models)]
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        temporal_perf = model_data.groupby('cutoff_date')[metric].mean()
        
        fig.add_trace(go.Scatter(
            x=temporal_perf.index,
            y=temporal_perf.values,
            mode='lines+markers',
            name=model,
            line=dict(color=colors[i], width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f'{metric} Performance Over Time',
        xaxis_title='Cutoff Date',
        yaxis_title=metric,
        height=500,
        legend=dict(x=1.05, y=1)
    )
    
    return fig

def create_horizon_comparison(results_df):
    """Create horizon comparison chart"""
    
    horizon_data = results_df.groupby(['model', 'forecast_horizon'])['MAPE'].mean().reset_index()
    
    fig = px.bar(
        horizon_data,
        x='model',
        y='MAPE',
        color='forecast_horizon',
        title='Model Performance by Forecast Horizon',
        labels={'MAPE': 'MAPE (%)', 'model': 'Model', 'forecast_horizon': 'Horizon (months)'},
        height=500
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def create_growth_capture_analysis(results_df):
    """Create growth capture analysis"""
    
    growth_data = results_df.groupby('model')['growth_capture_pct'].mean().sort_values(ascending=False)
    
    # Color based on performance
    colors = ['green' if x > 90 else 'orange' if x > 70 else 'red' for x in growth_data.values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=growth_data.index,
            y=growth_data.values,
            marker_color=colors,
            text=np.round(growth_data.values, 1),
            textposition='auto',
        )
    ])
    
    fig.add_hline(y=100, line_dash="dash", line_color="black", 
                  annotation_text="Perfect Capture")
    
    fig.update_layout(
        title='Growth Trend Capture by Model<br><sub>% of Actual Growth Captured</sub>',
        xaxis_title='Model',
        yaxis_title='Growth Capture (%)',
        height=500
    )
    
    return fig

def create_stability_scatter(results_df):
    """Create stability vs performance scatter plot"""
    
    stability_data = []
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        mean_mape = model_data['MAPE'].mean()
        std_mape = model_data['MAPE'].std()
        cv = std_mape / mean_mape if mean_mape > 0 else np.inf
        stability_score = max(0, 1 - cv)
        
        stability_data.append({
            'model': model,
            'mean_mape': mean_mape,
            'cv_mape': cv,
            'stability_score': stability_score
        })
    
    stability_df = pd.DataFrame(stability_data)
    
    fig = px.scatter(
        stability_df,
        x='mean_mape',
        y='cv_mape',
        color='stability_score',
        size='stability_score',
        hover_name='model',
        title='Model Performance vs Stability<br><sub>Lower-Left is Better</sub>',
        labels={'mean_mape': 'Mean MAPE (%)', 'cv_mape': 'Coefficient of Variation'},
        color_continuous_scale='RdYlGn',
        height=500
    )
    
    return fig

# Main app
def main():
    st.title("📊 MEDIS Automated Model Evaluation Dashboard")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Evaluation Configuration")
    
    # Load data
    monthly_sales, competitive_data, df = load_evaluation_data()
    
    if monthly_sales is None:
        st.error("Could not load data. Please check the data file.")
        return
    
    st.sidebar.success(f"✅ Data loaded: {len(monthly_sales)} months")
    st.sidebar.info(f"📅 Range: {monthly_sales.index[0].strftime('%Y-%m')} to {monthly_sales.index[-1].strftime('%Y-%m')}")
    
    # Configuration options
    st.sidebar.subheader("Models to Evaluate")
    available_models = ['Enhanced_LSTM', 'Transformer', 'Prophet', 'XGBoost', 'LSTM', 'Naive', 'Seasonal_Naive']
    
    models_to_test = st.sidebar.multiselect(
        "Select Models",
        available_models,
        default=['Enhanced_LSTM', 'Transformer', 'Prophet', 'XGBoost']
    )
    
    st.sidebar.subheader("Forecast Horizons")
    horizons = st.sidebar.multiselect(
        "Forecast Horizons (months)",
        [1, 3, 6, 12],
        default=[3, 6, 12]
    )
    
    st.sidebar.subheader("Training Configuration")
    min_train_months = st.sidebar.slider(
        "Minimum Training Months",
        min_value=24,
        max_value=48,
        value=36,
        help="Minimum number of months for training data"
    )
    
    # Run evaluation button
    if st.sidebar.button("🚀 Run Comprehensive Evaluation", type="primary"):
        if not models_to_test:
            st.error("Please select at least one model to evaluate.")
            return
        
        if not horizons:
            st.error("Please select at least one forecast horizon.")
            return
        
        # Run evaluation
        validator, results_df = run_evaluation(
            monthly_sales, competitive_data, models_to_test, horizons, min_train_months
        )
        
        # Store in session state
        st.session_state['validator'] = validator
        st.session_state['results_df'] = results_df
        st.session_state['evaluation_complete'] = True
        
        st.success("✅ Evaluation completed successfully!")
    
    # Display results if available
    if 'evaluation_complete' in st.session_state and st.session_state['evaluation_complete']:
        
        results_df = st.session_state['results_df']
        validator = st.session_state['validator']
        
        # Overview metrics
        st.header("📈 Evaluation Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Scenarios", len(results_df))
        
        with col2:
            st.metric("Models Tested", results_df['model'].nunique())
        
        with col3:
            st.metric("Date Range", f"{len(results_df['cutoff_date'].unique())} periods")
        
        with col4:
            best_model = results_df.groupby('model')['MAPE'].mean().idxmin()
            best_mape = results_df.groupby('model')['MAPE'].mean().min()
            st.metric("Best Model (MAPE)", best_model, f"{best_mape:.1f}%")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔥 Performance Heatmap", 
            "📈 Temporal Analysis", 
            "⏱️ Horizon Effects", 
            "📊 Growth Capture", 
            "⚖️ Stability Analysis"
        ])
        
        with tab1:
            st.subheader("Performance Heatmap")
            
            metric_col, _ = st.columns([1, 3])
            with metric_col:
                selected_metric = st.selectbox(
                    "Select Metric",
                    ['MAPE', 'RMSE', 'R2', 'Directional_Accuracy'],
                    index=0
                )
            
            fig_heatmap = create_interactive_heatmap(results_df, selected_metric)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Summary table
            st.subheader("Summary Statistics")
            summary_stats = results_df.groupby('model').agg({
                'MAPE': ['mean', 'std', 'min', 'max'],
                'R2': ['mean', 'std'],
                'growth_capture_pct': 'mean'
            }).round(2)
            
            summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
            st.dataframe(summary_stats)
        
        with tab2:
            st.subheader("Temporal Performance Analysis")
            
            metric_col, _ = st.columns([1, 3])
            with metric_col:
                temporal_metric = st.selectbox(
                    "Select Metric for Temporal Analysis",
                    ['MAPE', 'RMSE', 'R2', 'Directional_Accuracy'],
                    index=0,
                    key="temporal_metric"
                )
            
            fig_temporal = create_temporal_performance_chart(results_df, temporal_metric)
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Insights
            st.subheader("Temporal Insights")
            
            # Find most stable model
            stability_scores = {}
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                cv = model_data['MAPE'].std() / model_data['MAPE'].mean()
                stability_scores[model] = 1 - cv
            
            most_stable = max(stability_scores, key=stability_scores.get)
            
            st.info(f"🏆 Most temporally stable model: **{most_stable}**")
            
            # Performance trends
            recent_performance = results_df[results_df['cutoff_date'] >= results_df['cutoff_date'].quantile(0.7)]
            recent_best = recent_performance.groupby('model')['MAPE'].mean().idxmin()
            
            st.info(f"📈 Best recent performer: **{recent_best}**")
        
        with tab3:
            st.subheader("Forecast Horizon Effects")
            
            fig_horizon = create_horizon_comparison(results_df)
            st.plotly_chart(fig_horizon, use_container_width=True)
            
            # Horizon degradation analysis
            st.subheader("Horizon Degradation Analysis")
            
            horizon_degradation = []
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                horizon_perf = model_data.groupby('forecast_horizon')['MAPE'].mean()
                
                if len(horizon_perf) > 1:
                    degradation = horizon_perf.max() - horizon_perf.min()
                    best_horizon = horizon_perf.idxmin()
                    worst_horizon = horizon_perf.idxmax()
                    
                    horizon_degradation.append({
                        'Model': model,
                        'Best Horizon': f"{best_horizon}m",
                        'Worst Horizon': f"{worst_horizon}m",
                        'Degradation (MAPE %)': round(degradation, 2)
                    })
            
            if horizon_degradation:
                degradation_df = pd.DataFrame(horizon_degradation).sort_values('Degradation (MAPE %)')
                st.dataframe(degradation_df, use_container_width=True)
        
        with tab4:
            st.subheader("Growth Trend Capture Analysis")
            
            fig_growth = create_growth_capture_analysis(results_df)
            st.plotly_chart(fig_growth, use_container_width=True)
            
            # Growth capture insights
            st.subheader("Growth Capture Insights")
            
            growth_performance = results_df.groupby('model')['growth_capture_pct'].agg(['mean', 'std']).round(1)
            growth_performance.columns = ['Mean Capture (%)', 'Std Capture (%)']
            growth_performance = growth_performance.sort_values('Mean Capture (%)', ascending=False)
            
            st.dataframe(growth_performance)
            
            # Identify models that consistently capture growth
            good_growth_models = growth_performance[
                (growth_performance['Mean Capture (%)'] > 80) & 
                (growth_performance['Std Capture (%)'] < 50)
            ].index.tolist()
            
            if good_growth_models:
                st.success(f"🎯 Models with consistent growth capture: {', '.join(good_growth_models)}")
        
        with tab5:
            st.subheader("Model Stability Analysis")
            
            fig_stability = create_stability_scatter(results_df)
            st.plotly_chart(fig_stability, use_container_width=True)
            
            # Stability ranking
            st.subheader("Stability Ranking")
            
            stability_ranking = []
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                mean_mape = model_data['MAPE'].mean()
                std_mape = model_data['MAPE'].std()
                cv = std_mape / mean_mape if mean_mape > 0 else np.inf
                stability_score = max(0, 1 - cv)
                
                stability_ranking.append({
                    'Model': model,
                    'Mean MAPE (%)': round(mean_mape, 2),
                    'MAPE Std (%)': round(std_mape, 2),
                    'Coefficient of Variation': round(cv, 3),
                    'Stability Score': round(stability_score, 3)
                })
            
            stability_df = pd.DataFrame(stability_ranking).sort_values('Stability Score', ascending=False)
            st.dataframe(stability_df, use_container_width=True)
        
        # Download section
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Detailed Results (CSV)",
                data=csv_data,
                file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Generate summary report
            report = validator.generate_performance_report()
            
            import json
            report_json = json.dumps(report, indent=2, default=str)
            
            st.download_button(
                label="📋 Download Summary Report (JSON)",
                data=report_json,
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Configure your evaluation settings in the sidebar and click 'Run Comprehensive Evaluation' to start.")
        
        # Show data preview
        if monthly_sales is not None:
            st.header("📊 Data Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("MEDIS Sales Trend")
                fig = px.line(
                    x=monthly_sales.index,
                    y=monthly_sales.values,
                    title="MEDIS Monthly Sales",
                    labels={'x': 'Date', 'y': 'Sales'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Data Statistics")
                stats = {
                    'Total Months': len(monthly_sales),
                    'Date Range': f"{monthly_sales.index[0].strftime('%Y-%m')} to {monthly_sales.index[-1].strftime('%Y-%m')}",
                    'Mean Monthly Sales': f"{monthly_sales.mean():,.0f}",
                    'Total Sales': f"{monthly_sales.sum():,.0f}",
                    'Growth Rate': f"{((monthly_sales.iloc[-1] / monthly_sales.iloc[0]) ** (1/len(monthly_sales)) - 1) * 100:.1f}% monthly"
                }
                
                for key, value in stats.items():
                    st.metric(key, value)

if __name__ == "__main__":
    main() 