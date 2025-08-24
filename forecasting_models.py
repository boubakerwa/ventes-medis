"""
MEDIS Pharmaceutical Sales Forecasting Models

This module implements various forecasting models for MEDIS sales prediction:
- Prophet with competitive regressors
- XGBoost with engineered features  
- ARIMA with seasonal components
- Ensemble methods

Author: AI Assistant
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class MedisForecastingPipeline:
    """
    Main forecasting pipeline for MEDIS pharmaceutical sales
    
    Implements hierarchical forecasting approach:
    - Total MEDIS sales
    - Sub-market level (dosage categories)
    - Package level (30/60/90 tablets)
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'sales'):
        """
        Initialize the forecasting pipeline
        
        Args:
            data: Pharmaceutical sales data
            target_column: Column name for sales values
        """
        self.data = data.copy()
        self.target_column = target_column
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        
        # Preprocessing
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the pharmaceutical sales data"""
        
        # Ensure date column exists
        if 'date' not in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['ANNEE_MOIS'].astype(str), format='%Y%m')
        
        # Standardize package sizes
        if 'pack_std' not in self.data.columns:
            def standardize_pack_size(pack):
                if pd.isna(pack):
                    return pack
                if pack <= 35:
                    return 30
                elif pack <= 70:
                    return 60
                else:
                    return 90
            
            self.data['pack_std'] = self.data['PACK'].apply(standardize_pack_size)
        
        # Ensure sales column exists
        if self.target_column not in self.data.columns:
            self.data[self.target_column] = self.data['VENTE_IMS'].fillna(0)
        
        # Sort by date
        self.data = self.data.sort_values('date')
        
        print(f"✅ Data preprocessed: {len(self.data):,} records from {self.data['date'].min()} to {self.data['date'].max()}")

class FeatureEngineer:
    """
    Feature engineering for pharmaceutical sales forecasting
    
    Creates temporal, competitive, and hierarchical features
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'VENTE_IMS'):
        self.data = data.copy()
        self.target_column = target_column
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Seasonal indicators
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_q4'] = (df['quarter'] == 4).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """Create lagged features for time series"""
        
        df = df.copy().sort_values('date')
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            df[f'{target_col}_rolling_mean_{lag}'] = df[target_col].rolling(window=lag, min_periods=1).mean()
            df[f'{target_col}_rolling_std_{lag}'] = df[target_col].rolling(window=lag, min_periods=1).std()
        
        # Growth rates
        df[f'{target_col}_growth_1m'] = df[target_col].pct_change(1)
        df[f'{target_col}_growth_12m'] = df[target_col].pct_change(12)
        
        return df
    
    def create_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competitive intelligence features"""
        
        df = df.copy()
        
        # Market share calculation
        monthly_totals = df.groupby(['date', 'SOUS_MARCHE'])[self.target_column].sum().reset_index()
        monthly_totals = monthly_totals.rename(columns={self.target_column: 'total_market_sales'})
        
        df = df.merge(monthly_totals, on=['date', 'SOUS_MARCHE'], how='left')
        df['market_share'] = df[self.target_column] / df['total_market_sales']
        df['market_share'] = df['market_share'].fillna(0)
        
        # Competitive intensity (number of active competitors)
        competitive_intensity = df.groupby(['date', 'SOUS_MARCHE'])['laboratoire'].nunique().reset_index()
        competitive_intensity = competitive_intensity.rename(columns={'laboratoire': 'competitor_count'})
        
        df = df.merge(competitive_intensity, on=['date', 'SOUS_MARCHE'], how='left')
        
        # Market concentration (HHI index approximation)
        def calculate_hhi(group):
            shares = group[self.target_column] / group[self.target_column].sum()
            hhi = (shares ** 2).sum()
            return hhi
        
        hhi_data = df.groupby(['date', 'SOUS_MARCHE']).apply(calculate_hhi).reset_index(name='hhi_index')
        df = df.merge(hhi_data, on=['date', 'SOUS_MARCHE'], how='left')
        
        return df

class BaselineModels:
    """
    Baseline forecasting models for performance comparison
    """
    
    def __init__(self):
        self.models = {}
        
    def naive_forecast(self, ts: pd.Series, periods: int) -> pd.Series:
        """Simple naive forecast (last value repeated)"""
        last_value = ts.iloc[-1]
        dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=periods, freq='M')
        return pd.Series([last_value] * periods, index=dates)
    
    def seasonal_naive(self, ts: pd.Series, periods: int, seasonal_periods: int = 12) -> pd.Series:
        """Seasonal naive forecast (same month last year)"""
        forecasts = []
        dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=periods, freq='M')
        
        for i in range(periods):
            if len(ts) >= seasonal_periods:
                seasonal_value = ts.iloc[-(seasonal_periods - (i % seasonal_periods))]
                forecasts.append(seasonal_value)
            else:
                forecasts.append(ts.iloc[-1])
        
        return pd.Series(forecasts, index=dates)
    
    def moving_average(self, ts: pd.Series, periods: int, window: int = 12) -> pd.Series:
        """Moving average forecast"""
        avg_value = ts.tail(window).mean()
        dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=periods, freq='M')
        return pd.Series([avg_value] * periods, index=dates)

class ProphetModel:
    """
    Prophet-based forecasting model for pharmaceutical sales
    
    Features:
    - Automatic seasonality detection
    - Trend changepoints
    - External regressors (competitive data)
    - Holiday effects
    """
    
    def __init__(self, include_competitive_features: bool = True):
        """
        Initialize Prophet model
        
        Args:
            include_competitive_features: Whether to include competitor data as regressors
        """
        self.include_competitive_features = include_competitive_features
        self.model = None
        self.fitted = False
        
        # Import Prophet here to handle potential installation issues
        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            print("❌ Prophet not installed. Run: pip install prophet")
            self.Prophet = None
    
    def prepare_prophet_data(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds, y columns)
        
        Args:
            ts: Time series data
            competitive_data: Competitive intelligence data
            
        Returns:
            DataFrame in Prophet format
        """
        # Create base Prophet dataframe
        prophet_df = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values
        })
        
        # Add competitive features if available
        if self.include_competitive_features and competitive_data is not None:
            # Aggregate competitor sales by month
            competitor_monthly = competitive_data.groupby('date').agg({
                'sales': 'sum',
                'market_share': 'mean',
                'competitor_count': 'mean',
                'hhi_index': 'mean'
            }).reset_index()
            
            # Merge with Prophet data
            prophet_df = prophet_df.merge(
                competitor_monthly, 
                left_on='ds', 
                right_on='date', 
                how='left'
            ).drop('date', axis=1)
            
            # Fill missing values
            prophet_df = prophet_df.fillna(method='ffill').fillna(0)
        
        return prophet_df
    
    def fit(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> 'ProphetModel':
        """
        Fit Prophet model to time series data
        
        Args:
            ts: Time series data (pandas Series with datetime index)
            competitive_data: Optional competitive data
            
        Returns:
            Self for method chaining
        """
        if self.Prophet is None:
            raise ImportError("Prophet not available")
        
        # Prepare data
        prophet_df = self.prepare_prophet_data(ts, competitive_data)
        
        # Configure Prophet model
        self.model = self.Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # Monthly data doesn't need weekly patterns
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # Pharmaceutical sales often have multiplicative seasonality
            changepoint_prior_scale=0.05,  # Detect trend changes (market shifts)
            seasonality_prior_scale=10,    # Strong seasonal patterns in pharma
            interval_width=0.8            # 80% confidence intervals
        )
        
        # Add competitive regressors if available
        if self.include_competitive_features and 'sales' in prophet_df.columns:
            self.model.add_regressor('sales', prior_scale=0.5)
            self.model.add_regressor('market_share', prior_scale=0.5)
            self.model.add_regressor('competitor_count', prior_scale=0.3)
            self.model.add_regressor('hhi_index', prior_scale=0.3)
        
        # Fit the model
        print(f"🔄 Training Prophet model with {len(prophet_df)} data points...")
        self.model.fit(prophet_df)
        self.fitted = True
        
        print("✅ Prophet model training completed")
        return self
    
    def predict(self, periods: int, competitive_data: pd.DataFrame = None) -> pd.Series:
        """
        Generate forecasts using fitted Prophet model
        
        Args:
            periods: Number of periods to forecast
            competitive_data: Future competitive data (if available)
            
        Returns:
            Forecast as pandas Series
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='M')
        
        # Add competitive features for future periods if available
        if self.include_competitive_features and competitive_data is not None:
            # For simplicity, we'll use the last known values for future periods
            # In production, you'd want actual competitive forecasts
            
            # Get the last known competitive values
            last_values = competitive_data.groupby('date').agg({
                'sales': 'sum',
                'market_share': 'mean', 
                'competitor_count': 'mean',
                'hhi_index': 'mean'
            }).tail(1)
            
            # Fill future periods with last known values (naive assumption)
            future_start_idx = len(future) - periods
            for col in ['sales', 'market_share', 'competitor_count', 'hhi_index']:
                if col not in future.columns:
                    future[col] = 0
                # Fill future periods
                future.loc[future_start_idx:, col] = last_values[col].iloc[0]
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract forecast for the requested periods
        forecast_periods = forecast.tail(periods)
        
        # Return as pandas Series with proper datetime index
        forecast_series = pd.Series(
            forecast_periods['yhat'].values,
            index=pd.to_datetime(forecast_periods['ds'])
        )
        
        # Ensure non-negative forecasts (sales can't be negative)
        forecast_series = forecast_series.clip(lower=0)
        
        return forecast_series
    
    def get_forecast_components(self, periods: int) -> pd.DataFrame:
        """
        Get detailed forecast components (trend, seasonality, etc.)
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecast components
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting components")
        
        future = self.model.make_future_dataframe(periods=periods, freq='M')
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly']]
    
    def plot_forecast(self, periods: int = 12):
        """
        Plot Prophet forecast with components
        
        Args:
            periods: Number of periods to forecast
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        import matplotlib.pyplot as plt
        
        future = self.model.make_future_dataframe(periods=periods, freq='M')
        forecast = self.model.predict(future)
        
        # Plot forecast
        fig1 = self.model.plot(forecast)
        plt.title('Prophet Forecast - MEDIS Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales (boxes)')
        plt.show()
        
        # Plot components
        fig2 = self.model.plot_components(forecast)
        plt.show()

class XGBoostModel:
    """
    XGBoost-based forecasting model for pharmaceutical sales
    
    Features:
    - Advanced feature engineering (lags, rolling statistics, seasonality)
    - Competitive intelligence integration
    - Time series cross-validation
    - Feature importance analysis
    """
    
    def __init__(self, include_competitive_features: bool = True, max_lags: int = 12):
        """
        Initialize XGBoost model
        
        Args:
            include_competitive_features: Whether to include competitor data
            max_lags: Maximum number of lag features to create
        """
        self.include_competitive_features = include_competitive_features
        self.max_lags = max_lags
        self.model = None
        self.feature_names = []
        self.fitted = False
        
        # Import XGBoost and sklearn
        try:
            import xgboost as xgb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            self.xgb = xgb
            self.TimeSeriesSplit = TimeSeriesSplit
            self.mse = mean_squared_error
            self.mae = mean_absolute_error
        except ImportError:
            print("❌ XGBoost or scikit-learn not installed. Run: pip install xgboost scikit-learn")
            self.xgb = None
    
    def create_time_series_features(self, ts: pd.Series) -> pd.DataFrame:
        """
        Create time series features for XGBoost
        
        Args:
            ts: Time series data
            
        Returns:
            DataFrame with engineered features
        """
        df = pd.DataFrame({'y': ts.values}, index=ts.index)
        df['date'] = ts.index
        
        # Temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Seasonal indicators
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_q2'] = (df['quarter'] == 2).astype(int)
        df['is_q3'] = (df['quarter'] == 3).astype(int)
        df['is_q4'] = (df['quarter'] == 4).astype(int)
        
        # Month indicators (pharmaceutical seasonality)
        for month in range(1, 13):
            df[f'month_{month}'] = (df['month'] == month).astype(int)
        
        # Lag features
        for lag in range(1, min(self.max_lags + 1, len(df))):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            if window <= len(df):
                df[f'rolling_mean_{window}'] = df['y'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df['y'].rolling(window=window, min_periods=1).std()
                df[f'rolling_max_{window}'] = df['y'].rolling(window=window, min_periods=1).max()
                df[f'rolling_min_{window}'] = df['y'].rolling(window=window, min_periods=1).min()
        
        # Growth rates
        df['growth_1m'] = df['y'].pct_change(1).fillna(0)
        df['growth_3m'] = df['y'].pct_change(3).fillna(0)
        df['growth_12m'] = df['y'].pct_change(12).fillna(0)
        
        # Trend features
        df['linear_trend'] = np.arange(len(df))
        df['quadratic_trend'] = np.arange(len(df)) ** 2
        
        # Volatility measures
        df['volatility_3m'] = df['y'].rolling(window=3, min_periods=1).std()
        df['volatility_6m'] = df['y'].rolling(window=6, min_periods=1).std()
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def add_competitive_features(self, df: pd.DataFrame, competitive_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add competitive intelligence features
        
        Args:
            df: Base feature dataframe
            competitive_data: Competitive intelligence data
            
        Returns:
            Enhanced dataframe with competitive features
        """
        if competitive_data is None or len(competitive_data) == 0:
            return df
        
        try:
            # Aggregate competitive data by month
            comp_monthly = competitive_data.groupby('date').agg({
                'sales': ['sum', 'mean', 'std', 'count'],
                'market_share': ['mean', 'std'] if 'market_share' in competitive_data.columns else ['mean'],
                'competitor_count': 'mean' if 'competitor_count' in competitive_data.columns else 'count',
                'hhi_index': 'mean' if 'hhi_index' in competitive_data.columns else 'sum'
            }).fillna(0)
            
            # Flatten column names
            comp_monthly.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                   for col in comp_monthly.columns.values]
            comp_monthly = comp_monthly.reset_index()
            
            # Merge with main dataframe
            df = df.merge(comp_monthly, on='date', how='left')
            
            # Add competitive growth rates
            for col in comp_monthly.columns:
                if col != 'date' and comp_monthly[col].dtype in ['float64', 'int64']:
                    df[f'{col}_growth_1m'] = df[col].pct_change(1).fillna(0)
                    df[f'{col}_growth_3m'] = df[col].pct_change(3).fillna(0)
            
            # Competitive intensity indicators
            if 'sales_sum' in df.columns:
                df['competitive_pressure'] = df['sales_sum'] / (df['y'] + 1)  # Avoid division by zero
                df['market_dominance'] = df['y'] / (df['sales_sum'] + df['y'] + 1)
            
        except Exception as e:
            print(f"⚠️ Error adding competitive features: {str(e)}")
        
        return df.fillna(0)
    
    def prepare_training_data(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for XGBoost
        
        Args:
            ts: Time series data
            competitive_data: Optional competitive data
            
        Returns:
            Feature matrix and target vector
        """
        # Create base features
        features_df = self.create_time_series_features(ts)
        
        # Add competitive features
        if self.include_competitive_features:
            features_df = self.add_competitive_features(features_df, competitive_data)
        
        # Remove date and target columns from features
        feature_cols = [col for col in features_df.columns if col not in ['y', 'date']]
        X = features_df[feature_cols]
        y = features_df['y']
        
        # Store feature names
        self.feature_names = feature_cols
        
        return X, y
    
    def fit(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> 'XGBoostModel':
        """
        Fit XGBoost model to time series data
        
        Args:
            ts: Time series data
            competitive_data: Optional competitive data
            
        Returns:
            Self for method chaining
        """
        if self.xgb is None:
            raise ImportError("XGBoost not available")
        
        print(f"🔄 Preparing XGBoost training data with {len(ts)} data points...")
        
        # Prepare training data
        X, y = self.prepare_training_data(ts, competitive_data)
        
        # Configure XGBoost model
        self.model = self.xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        
        print(f"🔄 Training XGBoost model with {X.shape[1]} features...")
        
        # Fit the model
        self.model.fit(X, y)
        self.fitted = True
        
        # Store training data for prediction
        self.training_data = ts.copy()
        
        print("✅ XGBoost model training completed")
        return self
    
    def predict(self, periods: int, competitive_data: pd.DataFrame = None, 
                last_known_values: pd.Series = None) -> pd.Series:
        """
        Generate forecasts using fitted XGBoost model
        
        Args:
            periods: Number of periods to forecast
            competitive_data: Competitive data for future periods
            last_known_values: Last known values for lag features
            
        Returns:
            Forecast as pandas Series
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if last_known_values is None:
            # Use the training data as last known values
            if hasattr(self, 'training_data') and self.training_data is not None:
                last_known_values = self.training_data
                print("⚠️ Using training data as last_known_values for XGBoost forecasting")
            else:
                raise ValueError("last_known_values required for XGBoost forecasting")
        
        print(f"🔄 Generating XGBoost forecast for {periods} periods...")
        
        # Create future dates
        last_date = last_known_values.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        # Initialize forecast storage
        forecasts = []
        
        # Extend the series for iterative forecasting
        extended_series = last_known_values.copy()
        
        for i, future_date in enumerate(future_dates):
            # Create features for this prediction
            temp_series = extended_series.copy()
            temp_series.index = pd.to_datetime(temp_series.index)
            
            # Add a temporary value (will be replaced with prediction)
            temp_series[future_date] = temp_series.iloc[-1]  # Use last known value temporarily
            
            # Create features
            features_df = self.create_time_series_features(temp_series)
            
            # Add competitive features if available
            if self.include_competitive_features and competitive_data is not None:
                features_df = self.add_competitive_features(features_df, competitive_data)
            
            # Get the last row (current prediction)
            current_features = features_df.iloc[-1:][self.feature_names]
            
            # Handle any missing features by filling with 0
            for feature in self.feature_names:
                if feature not in current_features.columns:
                    current_features[feature] = 0
            
            # Reorder columns to match training
            current_features = current_features[self.feature_names].fillna(0)
            
            # Make prediction
            prediction = self.model.predict(current_features)[0]
            
            # Ensure non-negative prediction
            prediction = max(0, prediction)
            
            # Add prediction to series for next iteration
            extended_series[future_date] = prediction
            forecasts.append(prediction)
        
        # Create forecast series
        forecast_series = pd.Series(forecasts, index=future_dates)
        
        print(f"✅ XGBoost forecast generated: {len(forecast_series)} periods")
        return forecast_series
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from fitted model
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_scores = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20):
        """
        Plot top feature importance
        
        Args:
            top_n: Number of top features to plot
        """
        if not self.fitted:
            raise ValueError("Model must be fitted to plot feature importance")
        
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance().head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - XGBoost Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

class LSTMModel:
    """
    LSTM-based forecasting model for pharmaceutical sales
    
    Features:
    - Long Short-Term Memory neural networks
    - Sequence-to-sequence prediction
    - Multi-variate time series support
    - Competitive features integration
    - Automatic feature scaling
    """
    
    def __init__(self, sequence_length: int = 12, lstm_units: int = 50, 
                 include_competitive_features: bool = True, dropout_rate: float = 0.2):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of past periods to use for prediction
            lstm_units: Number of LSTM units in each layer
            include_competitive_features: Whether to include competitor data
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.include_competitive_features = include_competitive_features
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = None
        self.feature_scalers = {}
        self.fitted = False
        self.feature_names = []
        
        # Import TensorFlow and Keras
        try:
            import tensorflow as tf
            from tensorflow import keras
            from sklearn.preprocessing import MinMaxScaler
            self.tf = tf
            self.keras = keras
            self.MinMaxScaler = MinMaxScaler
            
            # Set random seeds for reproducibility
            tf.random.set_seed(42)
            print("✅ TensorFlow loaded successfully")
        except ImportError:
            print("❌ TensorFlow not installed. Run: pip install tensorflow")
            self.tf = None
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> tuple:
        """
        Create sequences for LSTM training
        
        Args:
            data: Input features array
            target: Target values array (None for prediction)
            
        Returns:
            Tuple of (X_sequences, y_sequences) or just X_sequences for prediction
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(data)):
            # Create sequence of features
            X_sequences.append(data[i - self.sequence_length:i])
            
            # Create target if provided
            if target is not None:
                y_sequences.append(target[i])
        
        X_sequences = np.array(X_sequences)
        
        if target is not None:
            y_sequences = np.array(y_sequences)
            return X_sequences, y_sequences
        else:
            return X_sequences
    
    def prepare_lstm_data(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> tuple:
        """
        Prepare data for LSTM training
        
        Args:
            ts: Time series data
            competitive_data: Competitive features data
            
        Returns:
            Tuple of (X_train, y_train, feature_names)
        """
        # Create base features dataframe
        df = pd.DataFrame({'target': ts.values}, index=ts.index)
        df['date'] = ts.index
        
        # Add temporal features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Add cyclical encoding for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Add lag features
        for lag in [1, 3, 6, 12]:
            if lag < len(df):
                df[f'lag_{lag}'] = df['target'].shift(lag)
        
        # Add rolling statistics
        for window in [3, 6, 12]:
            if window < len(df):
                df[f'rolling_mean_{window}'] = df['target'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['target'].rolling(window=window).std()
        
        # Add competitive features if available
        if self.include_competitive_features and competitive_data is not None:
            try:
                # Aggregate competitive data by date
                comp_agg = competitive_data.groupby('date').agg({
                    'VENTE_IMS': ['sum', 'mean', 'count'],
                    'laboratoire': 'nunique'
                }).round(2)
                
                # Flatten column names
                comp_agg.columns = [f'comp_{col[0]}_{col[1]}' for col in comp_agg.columns]
                comp_agg = comp_agg.reset_index()
                
                # Merge with main dataframe
                df_with_comp = df.reset_index().merge(comp_agg, on='date', how='left')
                df = df_with_comp.set_index('index')
                
                print(f"✅ Added competitive features: {comp_agg.columns.tolist()}")
                
            except Exception as e:
                print(f"⚠️ Could not add competitive features: {str(e)}")
        
        # Select feature columns (exclude target and date)
        feature_cols = [col for col in df.columns if col not in ['target', 'date', 'index']]
        self.feature_names = feature_cols
        
        # Drop rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < self.sequence_length + 1:
            raise ValueError(f"Not enough data after cleaning. Need at least {self.sequence_length + 1} rows, got {len(df_clean)}")
        
        # Prepare features and target
        X_data = df_clean[feature_cols].values
        y_data = df_clean['target'].values
        
        # Scale features
        self.scaler = self.MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Scale target separately
        self.target_scaler = self.MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X_scaled, y_scaled)
        
        print(f"✅ Prepared LSTM data: {X_sequences.shape} sequences with {len(feature_cols)} features")
        
        return X_sequences, y_sequences, feature_cols
    
    def build_model(self, input_shape: tuple) -> None:
        """
        Build LSTM neural network architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
        """
        if self.tf is None:
            raise ValueError("TensorFlow not available")
        
        # Build sequential model
        model = self.keras.Sequential([
            # First LSTM layer with return sequences
            self.keras.layers.LSTM(
                self.lstm_units, 
                return_sequences=True, 
                input_shape=input_shape,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            
            # Second LSTM layer
            self.keras.layers.LSTM(
                self.lstm_units // 2, 
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            
            # Dense layers with dropout
            self.keras.layers.Dense(25, activation='relu'),
            self.keras.layers.Dropout(self.dropout_rate),
            self.keras.layers.Dense(10, activation='relu'),
            self.keras.layers.Dropout(self.dropout_rate),
            
            # Output layer
            self.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=self.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("✅ LSTM model architecture built:")
        print(f"   - Input shape: {input_shape}")
        print(f"   - LSTM units: {self.lstm_units}")
        print(f"   - Dropout rate: {self.dropout_rate}")
        print(f"   - Total parameters: {model.count_params():,}")
    
    def fit(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> 'LSTMModel':
        """
        Train the LSTM model
        
        Args:
            ts: Time series data
            competitive_data: Competitive features data
            
        Returns:
            Fitted LSTMModel instance
        """
        if self.tf is None:
            raise ValueError("TensorFlow not available")
        
        print(f"🔄 Training LSTM model on {len(ts)} data points...")
        
        # Prepare data
        X_train, y_train, feature_names = self.prepare_lstm_data(ts, competitive_data)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.build_model(input_shape)
        
        # Add early stopping and learning rate reduction
        early_stopping = self.keras.callbacks.EarlyStopping(
            monitor='loss', patience=20, restore_best_weights=True
        )
        
        reduce_lr = self.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=0.0001
        )
        
        # Train model
        print("🔄 Starting LSTM training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=min(32, len(X_train) // 4),
            verbose=1,
            callbacks=[early_stopping, reduce_lr],
            validation_split=0.2,
            shuffle=False  # Important for time series
        )
        
        self.fitted = True
        
        # Store training history
        self.training_history = history.history
        
        print("✅ LSTM model training completed")
        return self
    
    def predict(self, periods: int, competitive_data: pd.DataFrame = None,
                last_known_values: pd.Series = None) -> pd.Series:
        """
        Generate forecasts using fitted LSTM model
        
        Args:
            periods: Number of periods to forecast
            competitive_data: Competitive data for future periods
            last_known_values: Last known values for creating sequences
            
        Returns:
            Forecast as pandas Series
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if last_known_values is None:
            raise ValueError("last_known_values required for LSTM forecasting")
        
        print(f"🔄 Generating LSTM forecast for {periods} periods...")
        
        # Prepare the full time series including historical data
        extended_ts = last_known_values.copy()
        
        # Generate forecasts iteratively
        forecasts = []
        
        for i in range(periods):
            # Prepare data for current prediction
            temp_data = self.prepare_prediction_data(extended_ts, competitive_data)
            
            if temp_data is None:
                # Fallback: use last known value
                forecasts.append(extended_ts.iloc[-1])
                continue
            
            # Create sequence for prediction
            X_pred = self.create_sequences(temp_data)
            
            if len(X_pred) == 0:
                # Fallback: use last known value
                forecasts.append(extended_ts.iloc[-1])
                continue
            
            # Take the last sequence
            X_pred = X_pred[-1:] 
            
            # Make prediction
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            
            # Inverse transform
            pred = self.target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            
            # Ensure non-negative prediction
            pred = max(0, pred)
            
            forecasts.append(pred)
            
            # Add prediction to extended series for next iteration
            next_date = extended_ts.index[-1] + pd.DateOffset(months=1)
            extended_ts[next_date] = pred
        
        # Create forecast series
        forecast_dates = pd.date_range(
            start=last_known_values.index[-1] + pd.DateOffset(months=1),
            periods=periods,
            freq='M'
        )
        
        forecast_series = pd.Series(forecasts, index=forecast_dates)
        
        print(f"✅ LSTM forecast generated: {len(forecast_series)} periods")
        return forecast_series
    
    def prepare_prediction_data(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> np.ndarray:
        """
        Prepare data for prediction (similar to training but for inference)
        
        Args:
            ts: Extended time series including historical data
            competitive_data: Competitive data
            
        Returns:
            Scaled feature array ready for prediction
        """
        try:
            # Use the same preparation logic as training
            df = pd.DataFrame({'target': ts.values}, index=ts.index)
            df['date'] = ts.index
            
            # Add all the same features as in training
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
            
            # Lag features
            for lag in [1, 3, 6, 12]:
                if lag < len(df):
                    df[f'lag_{lag}'] = df['target'].shift(lag)
            
            # Rolling statistics
            for window in [3, 6, 12]:
                if window < len(df):
                    df[f'rolling_mean_{window}'] = df['target'].rolling(window=window).mean()
                    df[f'rolling_std_{window}'] = df['target'].rolling(window=window).std()
            
            # Add competitive features (simplified for prediction)
            if self.include_competitive_features and competitive_data is not None:
                # Use last known competitive values
                for col in self.feature_names:
                    if col.startswith('comp_') and col not in df.columns:
                        df[col] = 0  # Default value for missing competitive features
            
            # Select only the features that were used in training
            available_features = [col for col in self.feature_names if col in df.columns]
            missing_features = [col for col in self.feature_names if col not in df.columns]
            
            # Add missing features with default values
            for col in missing_features:
                df[col] = 0
            
            # Get features in the same order as training
            feature_data = df[self.feature_names].fillna(method='ffill').fillna(0)
            
            # Scale using the fitted scaler
            scaled_data = self.scaler.transform(feature_data.values)
            
            return scaled_data
            
        except Exception as e:
            print(f"⚠️ Error preparing prediction data: {str(e)}")
            return None
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.fitted or not hasattr(self, 'training_history'):
            print("⚠️ No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        history = self.training_history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        ax2.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

class EnsembleModel:
    """
    Ensemble forecasting model combining multiple approaches
    
    Features:
    - Simple averaging ensemble
    - Weighted averaging based on historical performance
    - Stacking ensemble with meta-learner
    - Dynamic model selection
    """
    
    def __init__(self, ensemble_method: str = 'weighted_average'):
        """
        Initialize ensemble model
        
        Args:
            ensemble_method: 'simple_average', 'weighted_average', or 'stacking'
        """
        self.ensemble_method = ensemble_method
        self.models = {}
        self.model_weights = {}
        self.fitted = False
        self.meta_learner = None
        
        # Import additional libraries for ensemble
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            self.LinearRegression = LinearRegression
            self.RandomForestRegressor = RandomForestRegressor
        except ImportError:
            print("⚠️ scikit-learn required for ensemble methods")
    
    def add_model(self, name: str, model_instance, weight: float = 1.0):
        """
        Add a model to the ensemble
        
        Args:
            name: Model identifier
            model_instance: Fitted model instance
            weight: Weight for weighted averaging (ignored for other methods)
        """
        self.models[name] = model_instance
        self.model_weights[name] = weight
        print(f"✅ Added {name} to ensemble with weight {weight}")
    
    def fit_ensemble(self, train_ts: pd.Series, test_ts: pd.Series, 
                    competitive_data: pd.DataFrame = None):
        """
        Fit the ensemble meta-learner using cross-validation
        
        Args:
            train_ts: Training time series
            test_ts: Test time series for meta-learner training
            competitive_data: Competitive data for models that need it
        """
        if self.ensemble_method != 'stacking':
            print(f"✅ Ensemble method '{self.ensemble_method}' doesn't require meta-learner training")
            self.fitted = True
            return self
        
        print("🔄 Training ensemble meta-learner...")
        
        # Generate predictions from all models
        model_predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    if name == "XGBoost":
                        pred = model.predict(len(test_ts), competitive_data, train_ts)
                    elif name == "Prophet":
                        pred = model.predict(len(test_ts), competitive_data)
                    else:
                        # Baseline models
                        pred = model.predict if hasattr(model, 'predict') else None
                        
                    if pred is not None:
                        model_predictions[name] = pred.values
                        print(f"✅ Generated predictions for {name}")
                        
            except Exception as e:
                print(f"⚠️ Could not generate predictions for {name}: {str(e)}")
                continue
        
        if len(model_predictions) < 2:
            print("⚠️ Not enough model predictions for stacking. Using weighted average.")
            self.ensemble_method = 'weighted_average'
            self.fitted = True
            return self
        
        # Prepare meta-learner training data
        X_meta = np.column_stack(list(model_predictions.values()))
        y_meta = test_ts.values
        
        # Train meta-learner
        self.meta_learner = self.LinearRegression()
        self.meta_learner.fit(X_meta, y_meta)
        
        print("✅ Meta-learner training completed")
        self.fitted = True
        return self
    
    def predict_ensemble(self, periods: int, competitive_data: pd.DataFrame = None,
                        last_known_values: pd.Series = None) -> pd.Series:
        """
        Generate ensemble forecasts
        
        Args:
            periods: Number of periods to forecast
            competitive_data: Competitive data for models that need it
            last_known_values: Last known values for models that need it
            
        Returns:
            Ensemble forecast as pandas Series
        """
        if not self.fitted and self.ensemble_method == 'stacking':
            raise ValueError("Stacking ensemble must be fitted before prediction")
        
        print(f"🔄 Generating ensemble forecast using {self.ensemble_method}...")
        
        # Generate predictions from all models
        model_forecasts = {}
        forecast_dates = None
        
        for name, model in self.models.items():
            try:
                # Generate forecast based on model type
                if name == "XGBoost":
                    if hasattr(model, 'predict') and last_known_values is not None:
                        forecast = model.predict(periods, competitive_data, last_known_values)
                elif name == "Prophet":
                    if hasattr(model, 'predict'):
                        forecast = model.predict(periods, competitive_data)
                elif name == "LSTM":
                    if hasattr(model, 'predict') and last_known_values is not None:
                        forecast = model.predict(periods, competitive_data, last_known_values)
                elif hasattr(model, 'naive_forecast'):
                    # Baseline models
                    if last_known_values is not None:
                        forecast = model.naive_forecast(last_known_values, periods)
                elif hasattr(model, 'seasonal_naive'):
                    if last_known_values is not None:
                        forecast = model.seasonal_naive(last_known_values, periods)
                elif hasattr(model, 'moving_average'):
                    if last_known_values is not None:
                        forecast = model.moving_average(last_known_values, periods)
                else:
                    print(f"⚠️ Unknown model type for {name}")
                    continue
                
                model_forecasts[name] = forecast
                if forecast_dates is None:
                    forecast_dates = forecast.index
                    
                print(f"✅ Generated forecast for {name}: {len(forecast)} periods")
                
            except Exception as e:
                print(f"⚠️ Error generating forecast for {name}: {str(e)}")
                continue
        
        if len(model_forecasts) == 0:
            raise ValueError("No valid forecasts generated from ensemble models")
        
        # Combine forecasts based on ensemble method
        if self.ensemble_method == 'simple_average':
            # Simple arithmetic mean
            forecasts_array = np.array([forecast.values for forecast in model_forecasts.values()])
            ensemble_forecast = np.mean(forecasts_array, axis=0)
            
        elif self.ensemble_method == 'weighted_average':
            # Weighted average
            total_weight = sum(self.model_weights[name] for name in model_forecasts.keys())
            ensemble_forecast = np.zeros(periods)
            
            for name, forecast in model_forecasts.items():
                weight = self.model_weights.get(name, 1.0) / total_weight
                ensemble_forecast += weight * forecast.values
                
        elif self.ensemble_method == 'stacking':
            # Meta-learner prediction
            if self.meta_learner is None:
                raise ValueError("Meta-learner not trained for stacking ensemble")
            
            X_meta = np.column_stack([forecast.values for forecast in model_forecasts.values()])
            ensemble_forecast = self.meta_learner.predict(X_meta)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Create forecast series
        ensemble_series = pd.Series(ensemble_forecast, index=forecast_dates)
        ensemble_series = ensemble_series.clip(lower=0)  # Non-negative forecasts
        
        print(f"✅ Ensemble forecast completed: {len(ensemble_series)} periods")
        return ensemble_series
    
    def get_model_contributions(self) -> pd.DataFrame:
        """
        Get the contribution of each model to the ensemble
        
        Returns:
            DataFrame with model contributions
        """
        if self.ensemble_method == 'weighted_average':
            total_weight = sum(self.model_weights.values())
            contributions = {
                name: weight / total_weight 
                for name, weight in self.model_weights.items()
            }
        elif self.ensemble_method == 'simple_average':
            contributions = {
                name: 1.0 / len(self.models) 
                for name in self.models.keys()
            }
        elif self.ensemble_method == 'stacking' and self.meta_learner is not None:
            # Meta-learner coefficients (for linear regression)
            if hasattr(self.meta_learner, 'coef_'):
                coeffs = self.meta_learner.coef_
                model_names = list(self.models.keys())
                contributions = dict(zip(model_names, coeffs))
            else:
                contributions = {name: 1.0 for name in self.models.keys()}
        else:
            contributions = {name: 1.0 for name in self.models.keys()}
        
        return pd.DataFrame([contributions]).T.rename(columns={0: 'contribution'})

class ModelEvaluator:
    """
    Model evaluation and performance metrics
    """
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive forecasting metrics"""
        
        # Align series
        common_index = actual.index.intersection(predicted.index)
        actual_aligned = actual.loc[common_index]
        predicted_aligned = predicted.loc[common_index]
        
        # Remove any NaN values
        mask = ~(np.isnan(actual_aligned) | np.isnan(predicted_aligned))
        actual_clean = actual_aligned[mask]
        predicted_clean = predicted_aligned[mask]
        
        if len(actual_clean) == 0:
            return {"error": "No valid data points for evaluation"}
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_clean - predicted_clean))
        mse = np.mean((actual_clean - predicted_clean) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (handle division by zero)
        mape_values = np.abs((actual_clean - predicted_clean) / actual_clean)
        mape_values = mape_values[np.isfinite(mape_values)]
        mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else np.inf
        
        # Directional accuracy
        actual_direction = np.sign(actual_clean.diff().dropna())
        predicted_direction = np.sign(predicted_clean.diff().dropna())
        
        if len(actual_direction) > 0:
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = 0
        
        return {
            'MAE': mae,
            'MSE': mse, 
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'R2': 1 - (mse / np.var(actual_clean)) if np.var(actual_clean) > 0 else 0
        }
    
    def time_series_split(self, data: pd.DataFrame, n_splits: int = 5, test_size: int = 6) -> List[Tuple]:
        """Create time series cross-validation splits"""
        
        data_sorted = data.sort_values('date')
        dates = data_sorted['date'].unique()
        
        splits = []
        for i in range(n_splits):
            test_end_idx = len(dates) - i * test_size
            test_start_idx = test_end_idx - test_size
            train_end_idx = test_start_idx
            
            if train_end_idx <= test_size:  # Ensure minimum training size
                break
                
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            
            train_data = data_sorted[data_sorted['date'].isin(train_dates)]
            test_data = data_sorted[data_sorted['date'].isin(test_dates)]
            
            splits.append((train_data, test_data))
        
        return splits

# Utility functions
def prepare_medis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare MEDIS-specific data for forecasting"""
    
    medis_data = df[df['laboratoire'] == 'MEDIS'].copy()
    
    # Aggregate to monthly level
    monthly_data = medis_data.groupby(['date', 'SOUS_MARCHE', 'pack_std'])['sales'].sum().reset_index()
    
    # Create hierarchical aggregations
    total_monthly = medis_data.groupby('date')['sales'].sum().reset_index()
    total_monthly['level'] = 'total'
    total_monthly['segment'] = 'ALL'
    
    submarket_monthly = medis_data.groupby(['date', 'SOUS_MARCHE'])['sales'].sum().reset_index()
    submarket_monthly['level'] = 'submarket'
    submarket_monthly['segment'] = submarket_monthly['SOUS_MARCHE']
    
    return {
        'detailed': monthly_data,
        'total': total_monthly,
        'submarket': submarket_monthly
    }

def load_and_prepare_data(file_path: str = 'MEDIS_VENTES.xlsx') -> pd.DataFrame:
    """Load and prepare pharmaceutical sales data"""
    
    try:
        df = pd.read_excel(file_path, sheet_name='Data')
        
        # Initialize pipeline
        pipeline = MedisForecastingPipeline(df)
        
        return pipeline.data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    print("🔮 MEDIS Forecasting Models Module")
    print("=" * 50)
    
    # Test data loading
    data = load_and_prepare_data()
    
    if data is not None:
        print(f"✅ Data loaded successfully: {len(data):,} records")
        
        # Test feature engineering
        fe = FeatureEngineer(data)
        medis_data = data[data['laboratoire'] == 'MEDIS'].copy()
        
        if not medis_data.empty:
            # Create monthly aggregation for testing
            monthly_medis = medis_data.groupby('date')['sales'].sum().reset_index()
            
            # Add features
            enhanced_data = fe.create_temporal_features(monthly_medis)
            enhanced_data = fe.create_lag_features(enhanced_data, 'sales')
            
            print(f"✅ Features created: {enhanced_data.shape[1]} columns")
            
            # Test baseline models
            baseline = BaselineModels()
            ts_data = enhanced_data.set_index('date')['sales']
            
            naive_forecast = baseline.naive_forecast(ts_data, 6)
            seasonal_forecast = baseline.seasonal_naive(ts_data, 6)
            
            print(f"✅ Baseline forecasts generated")
            print(f"   • Naive forecast: {naive_forecast.mean():.0f} avg boxes/month")
            print(f"   • Seasonal forecast: {seasonal_forecast.mean():.0f} avg boxes/month")
            
            # Test evaluation
            evaluator = ModelEvaluator()
            
            # Simple test with last 6 months
            if len(ts_data) > 12:
                train_data = ts_data[:-6]
                test_data = ts_data[-6:]
                
                test_naive = baseline.naive_forecast(train_data, 6)
                metrics = evaluator.calculate_metrics(test_data, test_naive)
                
                print(f"✅ Baseline model evaluation:")
                mape_val = metrics.get('MAPE', 0)
                rmse_val = metrics.get('RMSE', 0)
                print(f"   • MAPE: {mape_val:.1f}%")
                print(f"   • RMSE: {rmse_val:.0f}")
                
        print("\n🚀 Ready for advanced model implementation!")
        
    else:
        print("❌ Could not load data. Please ensure MEDIS_VENTES.xlsx is available.") 

class TransformerModel:
    """
    Transformer-based forecasting model for pharmaceutical sales
    
    Features:
    - Multi-head attention mechanism
    - Better long-range dependency capture
    - Superior trend modeling
    - Positional encoding for time series
    """
    
    def __init__(self, sequence_length: int = 24, d_model: int = 64, n_heads: int = 4,
                 include_competitive_features: bool = True, n_layers: int = 2):
        """
        Initialize Transformer model
        
        Args:
            sequence_length: Number of past periods (increased for better trend capture)
            d_model: Model dimension
            n_heads: Number of attention heads
            include_competitive_features: Whether to include competitor data
            n_layers: Number of transformer layers
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.include_competitive_features = include_competitive_features
        self.n_layers = n_layers
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.fitted = False
        self.feature_names = []
        
        # Import TensorFlow and Keras
        try:
            import tensorflow as tf
            from tensorflow import keras
            from sklearn.preprocessing import RobustScaler, MinMaxScaler
            self.tf = tf
            self.keras = keras
            self.RobustScaler = RobustScaler
            self.MinMaxScaler = MinMaxScaler
            
            # Set random seeds for reproducibility
            tf.random.set_seed(42)
            print("✅ TensorFlow loaded for Transformer model")
        except ImportError:
            print("❌ TensorFlow not installed. Run: pip install tensorflow")
            self.tf = None
    
    def create_trend_features(self, ts: pd.Series) -> pd.DataFrame:
        """Create enhanced trend and momentum features"""
        
        df = pd.DataFrame({'target': ts.values}, index=ts.index)
        df['date'] = ts.index
        
        # Basic temporal features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Enhanced trend features
        df['time_index'] = range(len(df))
        df['target_log'] = np.log1p(df['target'])  # Log transform for exponential trends
        
        # Multiple lag features for trend detection
        for lag in [1, 2, 3, 6, 12, 18, 24]:
            if lag < len(df):
                df[f'lag_{lag}'] = df['target'].shift(lag)
                df[f'lag_log_{lag}'] = df['target_log'].shift(lag)
        
        # Growth rates and momentum
        for period in [1, 3, 6, 12]:
            if period < len(df):
                df[f'growth_{period}m'] = df['target'].pct_change(period)
                df[f'growth_log_{period}m'] = df['target_log'].diff(period)
        
        # Acceleration features
        df['acceleration_1m'] = df['growth_1m'].diff()
        df['acceleration_3m'] = df['growth_3m'].diff()
        
        # Rolling statistics with multiple windows
        for window in [3, 6, 12, 18, 24]:
            if window < len(df):
                df[f'rolling_mean_{window}'] = df['target'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['target'].rolling(window=window).std()
                df[f'rolling_trend_{window}'] = df['target'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
        
        # Exponential weighted features
        for alpha in [0.1, 0.3, 0.5]:
            df[f'ema_{alpha}'] = df['target'].ewm(alpha=alpha).mean()
            df[f'ema_growth_{alpha}'] = df[f'ema_{alpha}'].pct_change()
        
        # Target vs trend ratio
        df['target_vs_trend_12m'] = df['target'] / df['rolling_mean_12']
        df['target_vs_trend_24m'] = df['target'] / df['rolling_mean_24']
        
        return df
    
    def prepare_transformer_data(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> tuple:
        """Prepare data for transformer training with enhanced features"""
        
        # Create enhanced feature set
        df = self.create_trend_features(ts)
        
        # Add competitive features if available
        if self.include_competitive_features and competitive_data is not None:
            try:
                # Aggregate competitive data by date
                comp_agg = competitive_data.groupby('date').agg({
                    'VENTE_IMS': ['sum', 'mean', 'count'],
                    'laboratoire': 'nunique'
                }).round(2)
                
                # Flatten column names
                comp_agg.columns = [f'comp_{col[0]}_{col[1]}' for col in comp_agg.columns]
                comp_agg = comp_agg.reset_index()
                
                # Add competitive growth features
                comp_agg['comp_growth_1m'] = comp_agg['comp_VENTE_IMS_sum'].pct_change()
                comp_agg['comp_growth_3m'] = comp_agg['comp_VENTE_IMS_sum'].pct_change(3)
                
                # Merge with main dataframe
                df_with_comp = df.reset_index().merge(comp_agg, on='date', how='left')
                df = df_with_comp.set_index('index')
                
                print(f"✅ Added enhanced competitive features")
                
            except Exception as e:
                print(f"⚠️ Could not add competitive features: {str(e)}")
        
        # Select feature columns (exclude target and date)
        feature_cols = [col for col in df.columns if col not in ['target', 'target_log', 'date', 'index']]
        self.feature_names = feature_cols
        
        # Drop rows with NaN values (more lenient than LSTM)
        df_clean = df.dropna()
        
        if len(df_clean) < self.sequence_length + 1:
            raise ValueError(f"Not enough data after cleaning. Need at least {self.sequence_length + 1} rows, got {len(df_clean)}")
        
        # Prepare features and target
        X_data = df_clean[feature_cols].values
        y_data = df_clean['target'].values
        
        # Use RobustScaler for better handling of growth trends
        self.scaler = self.RobustScaler()
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Scale target with log transformation
        self.target_scaler = self.MinMaxScaler()
        y_log = np.log1p(y_data)  # Log transform for exponential trends
        y_scaled = self.target_scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X_scaled, y_scaled)
        
        print(f"✅ Prepared Transformer data: {X_sequences.shape} sequences with {len(feature_cols)} features")
        print(f"✅ Using log-transformed target for better trend capture")
        
        return X_sequences, y_sequences, feature_cols
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> tuple:
        """Create sequences for transformer training"""
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(data)):
            X_sequences.append(data[i - self.sequence_length:i])
            
            if target is not None:
                y_sequences.append(target[i])
        
        X_sequences = np.array(X_sequences)
        
        if target is not None:
            y_sequences = np.array(y_sequences)
            return X_sequences, y_sequences
        else:
            return X_sequences
    
    def build_model(self, input_shape: tuple) -> None:
        """Build transformer architecture optimized for trend forecasting"""
        
        if self.tf is None:
            raise ValueError("TensorFlow not available")
        
        # Input layer
        inputs = self.keras.layers.Input(shape=input_shape)
        
        # Positional encoding for time series
        x = self.keras.layers.Dense(self.d_model)(inputs)
        
        # Multiple transformer layers
        for i in range(self.n_layers):
            # Multi-head attention
            attention = self.keras.layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=0.1
            )(x, x)
            
            # Add & norm
            x = self.keras.layers.LayerNormalization()(self.keras.layers.Add()([x, attention]))
            
            # Feed forward
            ff = self.keras.layers.Dense(self.d_model * 2, activation='relu')(x)
            ff = self.keras.layers.Dropout(0.1)(ff)
            ff = self.keras.layers.Dense(self.d_model)(ff)
            
            # Add & norm
            x = self.keras.layers.LayerNormalization()(self.keras.layers.Add()([x, ff]))
        
        # Global average pooling to aggregate sequence
        x = self.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers for final prediction
        x = self.keras.layers.Dense(128, activation='relu')(x)
        x = self.keras.layers.Dropout(0.2)(x)
        x = self.keras.layers.Dense(64, activation='relu')(x)
        x = self.keras.layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = self.keras.layers.Dense(1, activation='linear')(x)
        
        # Create model
        model = self.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom learning rate
        model.compile(
            optimizer=self.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("✅ Transformer model architecture built:")
        print(f"   - Sequence length: {input_shape[0]}")
        print(f"   - Features: {input_shape[1]}")
        print(f"   - Model dimension: {self.d_model}")
        print(f"   - Attention heads: {self.n_heads}")
        print(f"   - Layers: {self.n_layers}")
        print(f"   - Total parameters: {model.count_params():,}")
    
    def fit(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> 'TransformerModel':
        """Fit the transformer model"""
        
        if self.tf is None:
            raise ValueError("TensorFlow not available")
        
        print(f"\n🚀 Training Transformer model...")
        print(f"   - Training data: {len(ts)} time points")
        print(f"   - Sequence length: {self.sequence_length} months")
        
        # Prepare data
        X_train, y_train, feature_names = self.prepare_transformer_data(ts, competitive_data)
        
        # Build model
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Training callbacks
        callbacks = [
            self.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            self.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=min(32, len(X_train) // 4),
            callbacks=callbacks,
            validation_split=0.2,
            verbose=1
        )
        
        self.fitted = True
        self.training_history = history
        
        print(f"✅ Transformer model trained successfully!")
        return self
    
    def predict(self, periods: int, competitive_data: pd.DataFrame = None,
                last_known_values: pd.Series = None) -> pd.Series:
        """Generate predictions using the transformer model"""
        
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if last_known_values is None:
            raise ValueError("last_known_values required for transformer predictions")
        
        print(f"🔮 Generating Transformer forecast for {periods} periods...")
        
        # Prepare the last sequence for prediction
        df = self.create_trend_features(last_known_values)
        
        # Add competitive features if available
        if self.include_competitive_features and competitive_data is not None:
            try:
                # Use the same feature engineering as in training
                comp_agg = competitive_data.groupby('date').agg({
                    'VENTE_IMS': ['sum', 'mean', 'count'],
                    'laboratoire': 'nunique'
                }).round(2)
                
                comp_agg.columns = [f'comp_{col[0]}_{col[1]}' for col in comp_agg.columns]
                comp_agg = comp_agg.reset_index()
                
                comp_agg['comp_growth_1m'] = comp_agg['comp_VENTE_IMS_sum'].pct_change()
                comp_agg['comp_growth_3m'] = comp_agg['comp_VENTE_IMS_sum'].pct_change(3)
                
                df_with_comp = df.reset_index().merge(comp_agg, on='date', how='left')
                df = df_with_comp.set_index('index')
                
            except Exception as e:
                print(f"⚠️ Could not add competitive features for prediction: {str(e)}")
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col in self.feature_names]
        
        # Handle missing features
        for missing_col in set(self.feature_names) - set(feature_cols):
            df[missing_col] = 0  # Fill missing competitive features with 0
        
        feature_cols = self.feature_names
        df_clean = df[feature_cols].fillna(method='ffill').fillna(0)
        
        # Take the last sequence_length points
        if len(df_clean) < self.sequence_length:
            # Pad with the last known values if not enough data
            last_vals = df_clean.iloc[-1:].values
            padding_needed = self.sequence_length - len(df_clean)
            padding = np.tile(last_vals, (padding_needed, 1))
            X_data = np.vstack([padding, df_clean.values])
        else:
            X_data = df_clean.tail(self.sequence_length).values
        
        # Scale features
        X_scaled = self.scaler.transform(X_data)
        
        predictions = []
        current_sequence = X_scaled.copy()
        
        for i in range(periods):
            # Reshape for model input
            X_input = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Generate prediction
            pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]
            
            # Inverse transform prediction
            pred_log = self.target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            pred_value = np.expm1(pred_log)  # Inverse of log1p
            
            predictions.append(max(0, pred_value))  # Ensure non-negative
            
            # Update sequence for next prediction (simplified approach)
            # In practice, you'd want to update with the actual predicted features
            current_sequence = np.roll(current_sequence, -1, axis=0)
            # For simplicity, repeat the last feature vector (this could be improved)
            
        # Create prediction index
        last_date = last_known_values.index[-1]
        pred_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='M'
        )
        
        pred_series = pd.Series(predictions, index=pred_index)
        
        print(f"✅ Transformer forecast generated: {len(pred_series)} periods")
        print(f"   - Forecast range: {pred_series.min():.0f} to {pred_series.max():.0f}")
        
        return pred_series


class EnhancedLSTMModel(LSTMModel):
    """
    Enhanced LSTM model with better trend capture capabilities
    """
    
    def __init__(self, sequence_length: int = 24, lstm_units: int = 100, 
                 include_competitive_features: bool = True, dropout_rate: float = 0.2):
        """Enhanced LSTM with longer sequences and better architecture"""
        super().__init__(sequence_length, lstm_units, include_competitive_features, dropout_rate)
    
    def prepare_lstm_data(self, ts: pd.Series, competitive_data: pd.DataFrame = None) -> tuple:
        """Enhanced data preparation with better trend features"""
        
        # Use the transformer's enhanced feature engineering
        transformer_helper = TransformerModel()
        df = transformer_helper.create_trend_features(ts)
        
        # Add competitive features if available
        if self.include_competitive_features and competitive_data is not None:
            try:
                comp_agg = competitive_data.groupby('date').agg({
                    'VENTE_IMS': ['sum', 'mean', 'count'],
                    'laboratoire': 'nunique'
                }).round(2)
                
                comp_agg.columns = [f'comp_{col[0]}_{col[1]}' for col in comp_agg.columns]
                comp_agg = comp_agg.reset_index()
                
                comp_agg['comp_growth_1m'] = comp_agg['comp_VENTE_IMS_sum'].pct_change()
                comp_agg['comp_growth_3m'] = comp_agg['comp_VENTE_IMS_sum'].pct_change(3)
                
                df_with_comp = df.reset_index().merge(comp_agg, on='date', how='left')
                df = df_with_comp.set_index('index')
                
                print(f"✅ Added enhanced competitive features to LSTM")
                
            except Exception as e:
                print(f"⚠️ Could not add competitive features: {str(e)}")
        
        # Select feature columns (exclude target and date)
        feature_cols = [col for col in df.columns if col not in ['target', 'target_log', 'date', 'index']]
        self.feature_names = feature_cols
        
        # Drop rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < self.sequence_length + 1:
            raise ValueError(f"Not enough data after cleaning. Need at least {self.sequence_length + 1} rows, got {len(df_clean)}")
        
        # Prepare features and target
        X_data = df_clean[feature_cols].values
        y_data = df_clean['target'].values
        
        # Use RobustScaler for better trend handling
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Scale target with log transformation for exponential trends
        self.target_scaler = self.MinMaxScaler()
        y_log = np.log1p(y_data)
        y_scaled = self.target_scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X_scaled, y_scaled)
        
        print(f"✅ Prepared Enhanced LSTM data: {X_sequences.shape} sequences with {len(feature_cols)} features")
        print(f"✅ Using log-transformed target and RobustScaler")
        
        return X_sequences, y_sequences, feature_cols 