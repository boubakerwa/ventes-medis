import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MEDIS Interactive Forecasting Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">💊 MEDIS Interactive Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown("### Dynamic Predictions with Confidence Intervals")

# Data loading function
@st.cache_data
def load_data():
    """Load and preprocess the pharmaceutical sales data"""
    try:
        df = pd.read_excel('MEDIS_VENTES.xlsx', sheet_name='Data')
        
        # Basic preprocessing
        df['date'] = pd.to_datetime(df['ANNEE_MOIS'].astype(str), format='%Y%m')
        df['sales'] = df['VENTE_IMS'].fillna(0)
        
        # Filter for MEDIS data and aggregate monthly
        medis_data = df[df['laboratoire'] == 'MEDIS']
        monthly_sales = medis_data.groupby('date')['sales'].sum().sort_index()
        
        # Get competitive data
        competitive_data = df[df['laboratoire'] != 'MEDIS'].groupby('date')['sales'].sum().sort_index()
        
        return monthly_sales, competitive_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load forecasting models
@st.cache_resource
def load_forecasting_models():
    """Load forecasting models"""
    try:
        from forecasting_models import BaselineModels, ProphetModel, XGBoostModel
        
        models = {
            'Naive': BaselineModels(),
            'Seasonal Naive': BaselineModels(),
            'Moving Average (6m)': BaselineModels(),
            'Prophet': ProphetModel(include_competitive_features=False),
            'XGBoost': XGBoostModel(include_competitive_features=False)
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

def calculate_confidence_intervals(ts_data, forecast, model_name, confidence_level=0.95):
    """
    Calculate confidence intervals for forecasts
    """
    try:
        # Calculate historical volatility
        historical_returns = ts_data.pct_change().dropna()
        volatility = historical_returns.std()
        
        # Handle case where volatility is NaN or very small
        if pd.isna(volatility) or volatility < 0.01:
            volatility = 0.15  # Default 15% volatility
        
        # Calculate confidence intervals based on historical volatility
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        
        # Create time-increasing volatility (uncertainty increases with time)
        time_factor = np.sqrt(np.arange(1, len(forecast) + 1))
        
        if model_name == 'Prophet':
            # Prophet has built-in uncertainty intervals
            error = forecast.values * volatility * z_score * time_factor
        else:
            # For other models, use scaled historical volatility
            base_error = forecast.mean() * volatility * z_score
            error = base_error * time_factor
        
        # Create confidence bounds as pandas Series with same index as forecast
        upper_bound = pd.Series(forecast.values + error, index=forecast.index)
        lower_bound = pd.Series(forecast.values - error, index=forecast.index)
        
        # Ensure non-negative bounds
        lower_bound = lower_bound.clip(lower=0)
        
        return lower_bound, upper_bound
        
    except Exception as e:
        # Return simple bounds if calculation fails
        st.warning(f"Could not calculate confidence intervals: {e}")
        simple_error = forecast.std() if len(forecast) > 1 else forecast.mean() * 0.1
        upper_bound = forecast + simple_error
        lower_bound = forecast - simple_error
        lower_bound = lower_bound.clip(lower=0)
        return lower_bound, upper_bound

def generate_forecast_with_confidence(model_name, model_instance, train_data, periods, confidence_level=0.95):
    """
    Generate forecast with confidence intervals
    """
    try:
        # Generate base forecast
        if model_name == 'Naive':
            forecast = model_instance.naive_forecast(train_data, periods)
        elif model_name == 'Seasonal Naive':
            forecast = model_instance.seasonal_naive(train_data, periods)
        elif model_name == 'Moving Average (6m)':
            forecast = model_instance.moving_average(train_data, periods, window=6)
        elif model_name == 'Prophet':
            model_instance.fit(train_data)
            forecast = model_instance.predict(periods)
        elif model_name == 'XGBoost':
            model_instance.fit(train_data)
            forecast = model_instance.predict(periods)
        else:
            return None, None, None
        
        # Ensure forecast has proper datetime index
        if forecast is not None and not forecast.empty:
            # Create proper future dates starting from the last training date
            last_date = train_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=periods,
                freq='M'
            )
            
            # Reindex forecast with proper dates if needed
            if len(forecast) == periods:
                forecast.index = future_dates
            
            # Calculate confidence intervals
            lower_bound, upper_bound = calculate_confidence_intervals(
                train_data, forecast, model_name, confidence_level
            )
            
            return forecast, lower_bound, upper_bound
        else:
            return None, None, None
        
    except Exception as e:
        st.error(f"Error generating forecast for {model_name}: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

def create_interactive_plot(historical_data, cutoff_date, forecast, lower_bound, upper_bound, model_name):
    """
    Create interactive Plotly visualization
    """
    fig = go.Figure()
    
    # Historical data
    historical_dates = historical_data.index
    historical_values = historical_data.values
    
    # Split historical data at cutoff
    train_mask = historical_dates <= cutoff_date
    test_mask = historical_dates > cutoff_date
    
    # Training data
    fig.add_trace(go.Scatter(
        x=historical_dates[train_mask],
        y=historical_values[train_mask],
        mode='lines+markers',
        name='Training Data',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Test data (if available)
    if np.any(test_mask):
        fig.add_trace(go.Scatter(
            x=historical_dates[test_mask],
            y=historical_values[test_mask],
            mode='lines+markers',
            name='Actual (Test)',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
    
    # Forecast
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines+markers',
            name=f'{model_name} Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        # Confidence intervals
        if lower_bound is not None and upper_bound is not None:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=upper_bound,
                mode='lines',
                name='Upper Bound (95%)',
                line=dict(color='rgba(255,0,0,0.3)', width=0),
                showlegend=False
            ))
            
            # Lower bound
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=lower_bound,
                mode='lines',
                name='Confidence Interval',
                line=dict(color='rgba(255,0,0,0.3)', width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)'
            ))
    
    # Add vertical line at cutoff
    fig.add_vline(
        x=cutoff_date, 
        line_dash="dot", 
        line_color="orange", 
        line_width=2,
        annotation_text="Cutoff Date"
    )
    
    # Customize layout
    fig.update_layout(
        title=f'{model_name} Forecast with 95% Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Sales (boxes)',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main application
def main():
    # Load data
    with st.spinner("Loading data..."):
        monthly_sales, competitive_data = load_data()
    
    if monthly_sales is None or monthly_sales.empty:
        st.error("Could not load MEDIS sales data. Please ensure MEDIS_VENTES.xlsx is available.")
        return
    
    # Load models
    models = load_forecasting_models()
    
    if not models:
        st.error("Could not load forecasting models.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Interactive Controls")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Forecasting Model",
        list(models.keys()),
        index=3  # Default to Prophet
    )
    
    # Cutoff date selection
    min_date = monthly_sales.index.min()
    max_date = monthly_sales.index.max() - pd.DateOffset(months=6)  # Leave room for forecast
    
    cutoff_date = st.sidebar.date_input(
        "Select Cutoff Date",
        value=pd.Timestamp('2023-06-01').date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        help="Data before this date will be used for training"
    )
    cutoff_date = pd.Timestamp(cutoff_date)
    
    # Forecast horizon
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (months)",
        min_value=3,
        max_value=24,
        value=12,
        help="Number of months to forecast into the future"
    )
    
    # Confidence level
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        [0.95, 0.99],
        format_func=lambda x: f"{x*100:.0f}%",
        help="Confidence level for prediction intervals"
    )
    
    # Real-time updates checkbox
    auto_update = st.sidebar.checkbox(
        "Auto-update predictions",
        value=True,
        help="Automatically update predictions when parameters change"
    )
    
    # Manual update button
    update_button = st.sidebar.button("🔄 Update Forecast", type="primary")
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    # Display key metrics
    train_data = monthly_sales[monthly_sales.index <= cutoff_date]
    test_data = monthly_sales[monthly_sales.index > cutoff_date]
    
    with col1:
        st.metric(
            "Training Data Points",
            len(train_data),
            help="Number of months used for training"
        )
    
    with col2:
        if len(train_data) > 0:
            latest_sales = train_data.iloc[-1]
            st.metric(
                "Latest Sales (Training)",
                f"{latest_sales:,.0f}",
                help="Most recent sales value in training data"
            )
        else:
            st.metric("Latest Sales", "N/A")
    
    with col3:
        if len(train_data) >= 12:
            yoy_growth = ((train_data.iloc[-1] / train_data.iloc[-13]) - 1) * 100
            st.metric(
                "YoY Growth (Training)",
                f"{yoy_growth:+.1f}%",
                help="Year-over-year growth rate"
            )
        else:
            st.metric("YoY Growth", "N/A")
    
    # Generate forecast
    if auto_update or update_button:
        if len(train_data) < 6:
            st.error("Insufficient training data. Please select a later cutoff date.")
            return
        
        with st.spinner(f"Generating {model_name} forecast..."):
            model_instance = models[model_name]
            
            forecast, lower_bound, upper_bound = generate_forecast_with_confidence(
                model_name, model_instance, train_data, forecast_horizon, confidence_level
            )
        
        if forecast is not None:
            # Create interactive plot
            fig = create_interactive_plot(
                monthly_sales, cutoff_date, forecast, lower_bound, upper_bound, model_name
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            st.subheader("📊 Forecast Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_forecast = forecast.mean()
                st.metric(
                    "Average Forecast",
                    f"{avg_forecast:,.0f}",
                    help="Average forecasted sales per month"
                )
            
            with col2:
                total_forecast = forecast.sum()
                st.metric(
                    "Total Forecast",
                    f"{total_forecast:,.0f}",
                    help="Total forecasted sales over the horizon"
                )
            
            with col3:
                if len(train_data) > 0:
                    vs_avg = ((forecast.mean() - train_data.mean()) / train_data.mean() * 100)
                    st.metric(
                        "vs Training Avg",
                        f"{vs_avg:+.1f}%",
                        help="Forecast vs training data average"
                    )
                else:
                    st.metric("vs Training Avg", "N/A")
            
            with col4:
                if len(test_data) > 0:
                    # Calculate MAPE if test data is available
                    test_forecast = forecast[:len(test_data)]
                    test_actual = test_data[:len(test_forecast)]
                    
                    if len(test_actual) > 0:
                        mape = np.mean(np.abs((test_actual - test_forecast) / test_actual)) * 100
                        st.metric(
                            "MAPE (Test)",
                            f"{mape:.1f}%",
                            help="Mean Absolute Percentage Error on test data"
                        )
                    else:
                        st.metric("MAPE", "N/A")
                else:
                    st.metric("MAPE", "No test data")
            
            # Detailed forecast table
            with st.expander("📋 Detailed Forecast Data"):
                forecast_df = pd.DataFrame({
                    'Date': forecast.index.strftime('%Y-%m'),
                    'Forecast': forecast.values.astype(int),
                    'Lower Bound': lower_bound.astype(int) if lower_bound is not None else None,
                    'Upper Bound': upper_bound.astype(int) if upper_bound is not None else None,
                    'Period': range(1, len(forecast) + 1)
                })
                
                st.dataframe(forecast_df, use_container_width=True)
                
                # Download button
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast Data",
                    data=csv,
                    file_name=f"medis_interactive_forecast_{model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Model insights
            st.subheader("🔍 Model Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Trend analysis
                if len(forecast) > 1:
                    trend = "📈 Increasing" if forecast.iloc[-1] > forecast.iloc[0] else "📉 Decreasing"
                    trend_rate = ((forecast.iloc[-1] / forecast.iloc[0]) - 1) * 100
                    
                    st.markdown(f"""
                    **Forecast Trend:** {trend}  
                    **Change Rate:** {trend_rate:+.1f}% over {forecast_horizon} months  
                    **Confidence Level:** {confidence_level*100:.0f}%  
                    **Model Type:** {model_name}
                    """)
            
            with col2:
                # Risk assessment
                if lower_bound is not None and upper_bound is not None:
                    uncertainty_range = (upper_bound.mean() - lower_bound.mean()) / forecast.mean() * 100
                    
                    risk_level = "🟢 Low" if uncertainty_range < 30 else "🟡 Medium" if uncertainty_range < 60 else "🔴 High"
                    
                    st.markdown(f"""
                    **Uncertainty Range:** ±{uncertainty_range:.1f}%  
                    **Risk Level:** {risk_level}  
                    **Prediction Quality:** {"Good" if uncertainty_range < 40 else "Moderate" if uncertainty_range < 70 else "Poor"}
                    """)
        else:
            st.error(f"Failed to generate forecast using {model_name}. Please try a different model or adjust the cutoff date.")
    
    # Instructions
    with st.expander("ℹ️ How to Use This Dashboard"):
        st.markdown("""
        **Interactive Features:**
        
        1. **Model Selection**: Choose from Naive, Seasonal Naive, Moving Average, Prophet, or XGBoost
        2. **Dynamic Cutoff**: Adjust the cutoff date to see how different training periods affect predictions
        3. **Forecast Horizon**: Control how far into the future to predict (3-24 months)
        4. **Confidence Intervals**: The shaded area shows prediction uncertainty (95% or 99%)
        5. **Real-time Updates**: Enable auto-update to see changes immediately
        
        **Plot Elements:**
        - **Blue line**: Training data (used to build the model)
        - **Green line**: Actual test data (if available after cutoff)
        - **Red dashed line**: Model predictions
        - **Red shaded area**: Confidence interval ("Schlauch")
        - **Orange dotted line**: Cutoff date
        
        **Tips:**
        - Try different cutoff dates to see temporal stability
        - Compare models by switching between them
        - Observe how confidence intervals widen over time
        - Use shorter horizons for better accuracy
        """)

if __name__ == "__main__":
    main() 