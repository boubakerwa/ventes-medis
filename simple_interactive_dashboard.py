import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MEDIS Simple Interactive Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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
        
        return monthly_sales
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Simple forecasting methods
def naive_forecast(ts_data, periods):
    """Simple naive forecast - repeat last value"""
    last_value = ts_data.iloc[-1]
    last_date = ts_data.index[-1]
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq='M'
    )
    
    # Create forecast
    forecast = pd.Series([last_value] * periods, index=future_dates)
    return forecast

def seasonal_naive_forecast(ts_data, periods, season_length=12):
    """Seasonal naive forecast"""
    last_date = ts_data.index[-1]
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq='M'
    )
    
    # Get seasonal pattern
    forecasts = []
    for i in range(periods):
        season_idx = i % season_length
        if len(ts_data) > season_idx:
            forecasts.append(ts_data.iloc[-(season_length - season_idx)])
        else:
            forecasts.append(ts_data.iloc[-1])
    
    forecast = pd.Series(forecasts, index=future_dates)
    return forecast

def moving_average_forecast(ts_data, periods, window=6):
    """Moving average forecast"""
    last_date = ts_data.index[-1]
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq='M'
    )
    
    # Calculate moving average
    if len(ts_data) >= window:
        ma_value = ts_data.tail(window).mean()
    else:
        ma_value = ts_data.mean()
    
    forecast = pd.Series([ma_value] * periods, index=future_dates)
    return forecast

def linear_trend_forecast(ts_data, periods):
    """Simple linear trend forecast"""
    last_date = ts_data.index[-1]
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq='M'
    )
    
    # Calculate linear trend from last 12 months
    recent_data = ts_data.tail(12) if len(ts_data) >= 12 else ts_data
    
    # Simple linear regression
    x = np.arange(len(recent_data))
    y = recent_data.values
    
    if len(x) > 1:
        slope = np.polyfit(x, y, 1)[0]
        last_value = recent_data.iloc[-1]
        
        # Project trend forward
        forecasts = []
        for i in range(1, periods + 1):
            forecast_value = last_value + (slope * i)
            forecasts.append(max(0, forecast_value))  # Ensure non-negative
        
        forecast = pd.Series(forecasts, index=future_dates)
    else:
        # Fallback to naive
        forecast = pd.Series([ts_data.iloc[-1]] * periods, index=future_dates)
    
    return forecast

def calculate_confidence_intervals(ts_data, forecast, confidence_level=0.95):
    """Calculate confidence intervals"""
    # Calculate historical volatility
    historical_returns = ts_data.pct_change().dropna()
    volatility = historical_returns.std()
    
    # Handle case where volatility is NaN
    if pd.isna(volatility) or volatility < 0.01:
        volatility = 0.15  # Default 15% volatility
    
    # Z-score for confidence level
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    
    # Time-increasing volatility
    time_factor = np.sqrt(np.arange(1, len(forecast) + 1))
    base_error = forecast.mean() * volatility * z_score
    error = base_error * time_factor
    
    # Create bounds
    upper_bound = pd.Series(forecast.values + error, index=forecast.index)
    lower_bound = pd.Series(forecast.values - error, index=forecast.index)
    lower_bound = lower_bound.clip(lower=0)
    
    return lower_bound, upper_bound

def create_interactive_plot(historical_data, cutoff_date, forecast, lower_bound, upper_bound, model_name):
    """Create interactive Plotly visualization"""
    fig = go.Figure()
    
    # Split historical data at cutoff
    train_data = historical_data[historical_data.index <= cutoff_date]
    test_data = historical_data[historical_data.index > cutoff_date]
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data.values,
        mode='lines+markers',
        name='Training Data',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Test data (if available)
    if not test_data.empty:
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines+markers',
            name='Actual (Test)',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
    
    # Forecast
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines+markers',
            name=f'{model_name} Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        # Confidence intervals (the "Schlauch")
        if lower_bound is not None and upper_bound is not None:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=upper_bound.values,
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(255,0,0,0.1)', width=0),
                showlegend=False
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=lower_bound.values,
                mode='lines',
                name='Confidence Interval',
                line=dict(color='rgba(255,0,0,0.1)', width=0),
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
        title=f'{model_name} Forecast with {confidence_level*100:.0f}% Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Sales (boxes)',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Main application
def main():
    # Load data
    with st.spinner("Loading data..."):
        monthly_sales = load_data()
    
    if monthly_sales is None or monthly_sales.empty:
        st.error("Could not load MEDIS sales data. Please ensure MEDIS_VENTES.xlsx is available.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Interactive Controls")
    
    # Model selection
    model_options = {
        'Naive': naive_forecast,
        'Seasonal Naive': seasonal_naive_forecast,
        'Moving Average (6m)': lambda ts, periods: moving_average_forecast(ts, periods, 6),
        'Moving Average (12m)': lambda ts, periods: moving_average_forecast(ts, periods, 12),
        'Linear Trend': linear_trend_forecast
    }
    
    model_name = st.sidebar.selectbox(
        "Select Forecasting Model",
        list(model_options.keys()),
        index=0
    )
    
    # Cutoff date selection
    min_date = monthly_sales.index.min()
    max_date = monthly_sales.index.max() - pd.DateOffset(months=3)
    
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
    
    # Auto-update checkbox
    auto_update = st.sidebar.checkbox(
        "Auto-update predictions",
        value=True,
        help="Automatically update predictions when parameters change"
    )
    
    # Manual update button
    update_button = st.sidebar.button("🔄 Update Forecast", type="primary")
    
    # Display key metrics
    train_data = monthly_sales[monthly_sales.index <= cutoff_date]
    test_data = monthly_sales[monthly_sales.index > cutoff_date]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Data Points", len(train_data))
    
    with col2:
        if len(train_data) > 0:
            st.metric("Latest Sales (Training)", f"{train_data.iloc[-1]:,.0f}")
        else:
            st.metric("Latest Sales", "N/A")
    
    with col3:
        if len(train_data) >= 12:
            yoy_growth = ((train_data.iloc[-1] / train_data.iloc[-13]) - 1) * 100
            st.metric("YoY Growth (Training)", f"{yoy_growth:+.1f}%")
        else:
            st.metric("YoY Growth", "N/A")
    
    # Generate forecast
    if auto_update or update_button:
        if len(train_data) < 3:
            st.error("Insufficient training data. Please select a later cutoff date.")
            return
        
        with st.spinner(f"Generating {model_name} forecast..."):
            try:
                # Generate forecast
                forecast_func = model_options[model_name]
                forecast = forecast_func(train_data, forecast_horizon)
                
                # Calculate confidence intervals
                lower_bound, upper_bound = calculate_confidence_intervals(
                    train_data, forecast, confidence_level
                )
                
                # Create interactive plot
                fig = create_interactive_plot(
                    monthly_sales, cutoff_date, forecast, lower_bound, upper_bound, model_name
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                st.subheader("📊 Forecast Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Forecast", f"{forecast.mean():,.0f}")
                
                with col2:
                    st.metric("Total Forecast", f"{forecast.sum():,.0f}")
                
                with col3:
                    vs_avg = ((forecast.mean() - train_data.mean()) / train_data.mean() * 100)
                    st.metric("vs Training Avg", f"{vs_avg:+.1f}%")
                
                with col4:
                    if len(test_data) > 0:
                        # Calculate MAPE if test data is available
                        test_periods = min(len(test_data), len(forecast))
                        if test_periods > 0:
                            test_actual = test_data.iloc[:test_periods]
                            test_forecast = forecast.iloc[:test_periods]
                            mape = np.mean(np.abs((test_actual - test_forecast) / test_actual)) * 100
                            st.metric("MAPE (Test)", f"{mape:.1f}%")
                        else:
                            st.metric("MAPE", "N/A")
                    else:
                        st.metric("MAPE", "No test data")
                
                # Detailed forecast table
                with st.expander("📋 Detailed Forecast Data"):
                    forecast_df = pd.DataFrame({
                        'Date': forecast.index.strftime('%Y-%m'),
                        'Forecast': forecast.values.astype(int),
                        'Lower Bound': lower_bound.values.astype(int),
                        'Upper Bound': upper_bound.values.astype(int),
                        'Period': range(1, len(forecast) + 1)
                    })
                    
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Download button
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Data",
                        data=csv,
                        file_name=f"medis_forecast_{model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Model insights
                st.subheader("🔍 Model Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    trend = "📈 Increasing" if forecast.iloc[-1] > forecast.iloc[0] else "📉 Decreasing"
                    trend_rate = ((forecast.iloc[-1] / forecast.iloc[0]) - 1) * 100
                    
                    st.markdown(f"""
                    **Forecast Trend:** {trend}  
                    **Change Rate:** {trend_rate:+.1f}% over {forecast_horizon} months  
                    **Confidence Level:** {confidence_level*100:.0f}%  
                    **Model Type:** {model_name}
                    """)
                
                with col2:
                    uncertainty_range = (upper_bound.mean() - lower_bound.mean()) / forecast.mean() * 100
                    risk_level = "🟢 Low" if uncertainty_range < 30 else "🟡 Medium" if uncertainty_range < 60 else "🔴 High"
                    
                    st.markdown(f"""
                    **Uncertainty Range:** ±{uncertainty_range:.1f}%  
                    **Risk Level:** {risk_level}  
                    **Prediction Quality:** {"Good" if uncertainty_range < 40 else "Moderate"}
                    """)
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Instructions
    with st.expander("ℹ️ How to Use This Dashboard"):
        st.markdown("""
        **Interactive Features:**
        
        1. **Model Selection**: Choose from different forecasting approaches
        2. **Dynamic Cutoff**: Adjust the cutoff date to see how different training periods affect predictions
        3. **Forecast Horizon**: Control how far into the future to predict (3-24 months)
        4. **Confidence Intervals**: The shaded area shows prediction uncertainty ("Schlauch")
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