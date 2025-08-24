#!/usr/bin/env python3
"""
Comprehensive Automated Model Evaluation for MEDIS

This script includes ALL sophisticated models:
- Prophet with competitive features
- XGBoost with feature engineering (FIXED)
- LSTM deep learning models
- Enhanced LSTM with advanced features
- Transformer with attention mechanism
- Ensemble methods

Usage:
    python comprehensive_evaluation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all models
try:
    from forecasting_models import (
        load_and_prepare_data,
        BaselineModels,
        ProphetModel,
        XGBoostModel,
        LSTMModel,
        ModelEvaluator,
        FeatureEngineer
    )
except ImportError as e:
    print(f"Error importing forecasting models: {e}")
    exit(1)

def calculate_metrics(actual, predicted):
    """
    Calculate comprehensive evaluation metrics
    """
    # Convert to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'MAPE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Directional_Accuracy': np.nan,
            'Growth_Capture': np.nan
        }
    
    # Calculate metrics
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Directional accuracy
    if len(actual) > 1:
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    else:
        directional_accuracy = np.nan
    
    # Growth capture (how well model captures growth trends)
    if len(actual) > 1:
        actual_growth = (actual[-1] / actual[0] - 1) * 100 if actual[0] > 0 else 0
        predicted_growth = (predicted[-1] / predicted[0] - 1) * 100 if predicted[0] > 0 else 0
        growth_capture = (predicted_growth / actual_growth) * 100 if actual_growth != 0 else 100
    else:
        growth_capture = np.nan
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Growth_Capture': growth_capture
    }

class WalkForwardValidator:
    """
    Walk-forward validation for time series models
    """
    
    def __init__(self, cutoff_dates, horizons):
        self.cutoff_dates = cutoff_dates
        self.horizons = horizons
        self.results = []
        
    def evaluate_model(self, model_name, model_instance, data, competitive_data=None):
        """
        Evaluate a model across all cutoff dates and horizons
        """
        print(f"\n🔄 Evaluating {model_name}...")
        
        for cutoff_date in self.cutoff_dates:
            print(f"  📅 Cutoff: {cutoff_date.strftime('%Y-%m')}")
            
            # Split data
            train_data = data[data.index <= cutoff_date]
            
            if len(train_data) < 12:  # Need at least 12 months for training
                continue
                
            try:
                # Fit model
                print(f"    🔄 Training {model_name}...")
                
                if model_name in ['Prophet', 'XGBoost', 'LSTM', 'Enhanced_LSTM', 'Transformer']:
                    if competitive_data is not None:
                        train_competitive = competitive_data[competitive_data.index <= cutoff_date]
                        model_instance.fit(train_data, train_competitive)
                    else:
                        model_instance.fit(train_data)
                else:
                    # Baseline models don't need fitting
                    pass
                
                # Test multiple horizons
                for horizon in self.horizons:
                    test_end = cutoff_date + pd.DateOffset(months=horizon)
                    test_data = data[(data.index > cutoff_date) & (data.index <= test_end)]
                    
                    if len(test_data) == 0:
                        continue
                    
                    print(f"    🎯 Horizon: {horizon}m, Test points: {len(test_data)}")
                    
                    try:
                        # Generate predictions
                        if model_name == 'Naive':
                            prediction = model_instance.naive_forecast(train_data, len(test_data))
                        elif model_name == 'Seasonal_Naive':
                            prediction = model_instance.seasonal_naive(train_data, len(test_data))
                        elif model_name == 'Moving_Average_6m':
                            prediction = model_instance.moving_average(train_data, len(test_data), window=6)
                        elif model_name == 'Moving_Average_12m':
                            prediction = model_instance.moving_average(train_data, len(test_data), window=12)
                        elif model_name in ['Prophet', 'XGBoost', 'LSTM', 'Enhanced_LSTM', 'Transformer']:
                            # Advanced models - pass train_data as last_known_values
                            if model_name == 'XGBoost':
                                # XGBoost needs last_known_values but we fixed it to use training data
                                prediction = model_instance.predict(len(test_data))
                            else:
                                prediction = model_instance.predict(len(test_data))
                        else:
                            print(f"    ⚠️ Model {model_name} not implemented yet")
                            continue
                        
                        # Align prediction with test data
                        if len(prediction) > len(test_data):
                            prediction = prediction[:len(test_data)]
                        elif len(prediction) < len(test_data):
                            # Extend prediction if needed
                            last_value = prediction.iloc[-1] if len(prediction) > 0 else train_data.iloc[-1]
                            missing = len(test_data) - len(prediction)
                            extension = pd.Series([last_value] * missing, 
                                               index=test_data.index[-missing:])
                            prediction = pd.concat([prediction, extension])
                        
                        # Calculate metrics
                        metrics = calculate_metrics(test_data.values, prediction.values)
                        
                        # Store results
                        result = {
                            'model': model_name,
                            'cutoff_date': cutoff_date,
                            'horizon': horizon,
                            'train_size': len(train_data),
                            'test_size': len(test_data),
                            **metrics
                        }
                        
                        self.results.append(result)
                        
                        print(f"    ✅ MAPE: {metrics['MAPE']:.1f}%, RMSE: {metrics['RMSE']:.0f}")
                        
                    except Exception as e:
                        print(f"    ❌ Error predicting {model_name} for horizon {horizon}: {e}")
                        continue
                        
            except Exception as e:
                print(f"  ❌ Error training {model_name}: {e}")
                continue
    
    def get_results_dataframe(self):
        """
        Convert results to DataFrame
        """
        return pd.DataFrame(self.results)

def create_evaluation_dashboard(results_df, save_path=None):
    """
    Create comprehensive evaluation dashboard
    """
    if results_df.empty:
        print("❌ No results to plot")
        return
    
    # Clean results - remove rows with NaN MAPE
    clean_df = results_df.dropna(subset=['MAPE'])
    
    if clean_df.empty:
        print("❌ No valid results to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('MEDIS Comprehensive Model Evaluation Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Performance Heatmap
    try:
        pivot_mape = clean_df.pivot_table(values='MAPE', index='model', columns='horizon', aggfunc='mean')
        if not pivot_mape.empty:
            sns.heatmap(pivot_mape, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                       ax=axes[0, 0], cbar_kws={'label': 'MAPE (%)'})
            axes[0, 0].set_title('Performance Heatmap (MAPE)', fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No data for heatmap', ha='center', va='center')
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 2. Model Performance Distribution
    try:
        if len(clean_df['model'].unique()) > 0:
            model_performance = clean_df.groupby('model')['MAPE'].agg(['mean', 'std']).reset_index()
            model_performance = model_performance.sort_values('mean')
            
            x_pos = range(len(model_performance))
            axes[0, 1].bar(x_pos, model_performance['mean'], 
                          yerr=model_performance['std'], capsize=5, alpha=0.7)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(model_performance['model'], rotation=45, ha='right')
            axes[0, 1].set_title('Model Performance (MAPE ± Std)', fontweight='bold')
            axes[0, 1].set_ylabel('MAPE (%)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No models to compare', ha='center', va='center')
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 3. Horizon Effects
    try:
        if len(clean_df['horizon'].unique()) > 1:
            horizon_perf = clean_df.groupby('horizon')['MAPE'].agg(['mean', 'std']).reset_index()
            horizon_perf = horizon_perf.sort_values('horizon')
            
            axes[0, 2].errorbar(horizon_perf['horizon'], horizon_perf['mean'], 
                               yerr=horizon_perf['std'], marker='o', capsize=5)
            axes[0, 2].set_title('Horizon Effects on Performance', fontweight='bold')
            axes[0, 2].set_xlabel('Forecast Horizon (months)')
            axes[0, 2].set_ylabel('MAPE (%)')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Single horizon only', ha='center', va='center')
    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 4. Temporal Stability
    try:
        if len(clean_df['cutoff_date'].unique()) > 1:
            temporal_perf = clean_df.groupby('cutoff_date')['MAPE'].agg(['mean', 'std']).reset_index()
            temporal_perf['cutoff_str'] = temporal_perf['cutoff_date'].dt.strftime('%Y-%m')
            
            x_pos = range(len(temporal_perf))
            axes[1, 0].bar(x_pos, temporal_perf['mean'], 
                          yerr=temporal_perf['std'], capsize=5, alpha=0.7)
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(temporal_perf['cutoff_str'], rotation=45)
            axes[1, 0].set_title('Temporal Performance Stability', fontweight='bold')
            axes[1, 0].set_ylabel('MAPE (%)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Single period only', ha='center', va='center')
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 5. R² vs MAPE Scatter
    try:
        clean_r2 = clean_df.dropna(subset=['R2'])
        if not clean_r2.empty:
            for model in clean_r2['model'].unique():
                model_data = clean_r2[clean_r2['model'] == model]
                axes[1, 1].scatter(model_data['MAPE'], model_data['R2'], 
                                  label=model, alpha=0.7, s=50)
            
            axes[1, 1].set_xlabel('MAPE (%)')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].set_title('Accuracy vs Fit Quality', fontweight='bold')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No R² data available', ha='center', va='center')
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 6. Growth Capture Analysis
    try:
        clean_growth = clean_df.dropna(subset=['Growth_Capture'])
        if not clean_growth.empty:
            # Filter reasonable growth capture values
            clean_growth = clean_growth[(clean_growth['Growth_Capture'] >= -200) & 
                                      (clean_growth['Growth_Capture'] <= 300)]
            
            if not clean_growth.empty:
                growth_by_model = clean_growth.groupby('model')['Growth_Capture'].mean().sort_values()
                
                x_pos = range(len(growth_by_model))
                axes[1, 2].bar(x_pos, growth_by_model.values, alpha=0.7)
                axes[1, 2].set_xticks(x_pos)
                axes[1, 2].set_xticklabels(growth_by_model.index, rotation=45, ha='right')
                axes[1, 2].set_title('Growth Capture by Model', fontweight='bold')
                axes[1, 2].set_ylabel('Growth Capture (%)')
                axes[1, 2].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Perfect Capture')
                axes[1, 2].legend()
            else:
                axes[1, 2].text(0.5, 0.5, 'No valid growth data', ha='center', va='center')
        else:
            axes[1, 2].text(0.5, 0.5, 'No growth capture data', ha='center', va='center')
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 7. Directional Accuracy
    try:
        clean_dir = clean_df.dropna(subset=['Directional_Accuracy'])
        if not clean_dir.empty:
            dir_by_model = clean_dir.groupby('model')['Directional_Accuracy'].mean().sort_values(ascending=False)
            
            x_pos = range(len(dir_by_model))
            axes[2, 0].bar(x_pos, dir_by_model.values, alpha=0.7, color='lightblue')
            axes[2, 0].set_xticks(x_pos)
            axes[2, 0].set_xticklabels(dir_by_model.index, rotation=45, ha='right')
            axes[2, 0].set_title('Directional Accuracy by Model', fontweight='bold')
            axes[2, 0].set_ylabel('Directional Accuracy (%)')
            axes[2, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random')
            axes[2, 0].legend()
        else:
            axes[2, 0].text(0.5, 0.5, 'No directional data', ha='center', va='center')
    except Exception as e:
        axes[2, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 8. Model Stability (CV of MAPE)
    try:
        stability_data = []
        for model in clean_df['model'].unique():
            model_mapes = clean_df[clean_df['model'] == model]['MAPE']
            if len(model_mapes) > 1:
                cv = model_mapes.std() / model_mapes.mean() if model_mapes.mean() > 0 else np.nan
                stability_data.append({'model': model, 'cv': cv, 'mean_mape': model_mapes.mean()})
        
        if stability_data:
            stability_df = pd.DataFrame(stability_data).dropna()
            if not stability_df.empty:
                stability_df = stability_df.sort_values('cv')
                
                x_pos = range(len(stability_df))
                axes[2, 1].bar(x_pos, stability_df['cv'], alpha=0.7, color='lightgreen')
                axes[2, 1].set_xticks(x_pos)
                axes[2, 1].set_xticklabels(stability_df['model'], rotation=45, ha='right')
                axes[2, 1].set_title('Model Stability (Lower = More Stable)', fontweight='bold')
                axes[2, 1].set_ylabel('Coefficient of Variation')
            else:
                axes[2, 1].text(0.5, 0.5, 'No stability data', ha='center', va='center')
        else:
            axes[2, 1].text(0.5, 0.5, 'Insufficient data for stability', ha='center', va='center')
    except Exception as e:
        axes[2, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 9. Summary Statistics Table
    try:
        summary_stats = clean_df.groupby('model').agg({
            'MAPE': ['mean', 'std', 'min', 'max'],
            'RMSE': 'mean',
            'R2': 'mean'
        }).round(2)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        
        # Create text summary
        axes[2, 2].axis('off')
        summary_text = "Model Performance Summary\n\n"
        
        if not summary_stats.empty:
            for model in summary_stats.index[:5]:  # Top 5 models
                mape_mean = summary_stats.loc[model, 'MAPE_mean']
                mape_std = summary_stats.loc[model, 'MAPE_std']
                summary_text += f"{model}: {mape_mean:.1f}% ± {mape_std:.1f}%\n"
        else:
            summary_text += "No summary statistics available"
        
        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[2, 2].set_title('Performance Summary', fontweight='bold')
        
    except Exception as e:
        axes[2, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """
    Main evaluation function
    """
    print("🚀 Starting Comprehensive MEDIS Model Evaluation")
    print("=" * 60)
    
    # Load data
    print("📊 Loading MEDIS data...")
    try:
        df = load_and_prepare_data('MEDIS_VENTES.xlsx')
        print(f"✅ Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Filter for MEDIS data
    medis_data = df[df['laboratoire'] == 'MEDIS'] if 'laboratoire' in df.columns else df
    
    if medis_data.empty:
        print("❌ No MEDIS data found")
        return
    
    # Aggregate monthly sales
    monthly_sales = medis_data.groupby(medis_data.index)['sales'].sum().sort_index()
    print(f"📈 MEDIS monthly data: {len(monthly_sales)} months")
    
    # Prepare competitive data
    competitive_data = None
    if 'laboratoire' in df.columns:
        try:
            competitive_data = df[df['laboratoire'] != 'MEDIS'].groupby(df.index)['sales'].sum().sort_index()
            print(f"🏢 Competitive data: {len(competitive_data)} months")
        except Exception as e:
            print(f"⚠️ Could not prepare competitive data: {e}")
    
    # Define evaluation parameters
    cutoff_dates = [
        pd.Timestamp('2021-11-01'),
        pd.Timestamp('2022-11-01'), 
        pd.Timestamp('2023-11-01')
    ]
    
    forecast_horizons = [3, 6, 12]  # months
    
    print(f"\n🔬 Testing ALL sophisticated models across {len(cutoff_dates)} periods")
    print(f"   Cutoff dates: {[d.strftime('%Y-%m') for d in cutoff_dates]}")
    print(f"   Horizons: {forecast_horizons} months")
    
    # Initialize models
    print(f"\n🤖 Initializing models...")
    
    models_config = {
        'Naive': {
            'class': BaselineModels,
            'method': 'naive_forecast',
            'requires_competitive': False
        },
        'Seasonal_Naive': {
            'class': BaselineModels,
            'method': 'seasonal_naive',
            'requires_competitive': False
        },
        'Moving_Average_6m': {
            'class': BaselineModels,
            'method': 'moving_average',
            'params': {'window': 6},
            'requires_competitive': False
        },
        'Prophet': {
            'class': ProphetModel,
            'params': {'include_competitive_features': True},
            'requires_competitive': True
        },
        'XGBoost': {
            'class': XGBoostModel,
            'params': {'include_competitive_features': True, 'max_lags': 6},
            'requires_competitive': True
        },
        'LSTM': {
            'class': LSTMModel,
            'params': {
                'sequence_length': 12,
                'lstm_units': 50,
                'include_competitive_features': True,
                'dropout_rate': 0.2
            },
            'requires_competitive': True
        }
    }
    
    results = []
    evaluator = ModelEvaluator()
    
    # Run evaluations
    total_scenarios = len(cutoff_dates) * len(forecast_horizons) * len(models_config)
    current_scenario = 0
    
    for cutoff_date in cutoff_dates:
        print(f"\n📅 Evaluating cutoff: {cutoff_date.strftime('%Y-%m')}")
        
        # Check if cutoff date exists in data
        if cutoff_date not in monthly_sales.index:
            print(f"   ⚠️  Skipping {cutoff_date} - not in data")
            continue
        
        # Prepare competitive features for this cutoff
        print(f"   🔧 Preparing competitive features...")
        competitive_features = prepare_competitive_features(df, cutoff_date)
            
        for horizon in forecast_horizons:
            print(f"  🎯 Horizon: {horizon} months")
            
            # Prepare train/test split
            cutoff_idx = monthly_sales.index.get_loc(cutoff_date)
            
            # Check if we have enough data
            if cutoff_idx + horizon >= len(monthly_sales):
                print(f"    ⚠️  Not enough data for {horizon}m horizon")
                continue
            
            train_data = monthly_sales[:cutoff_idx + 1]
            test_data = monthly_sales[cutoff_idx + 1:cutoff_idx + 1 + horizon]
            
            print(f"    📊 Train: {len(train_data)} months, Test: {len(test_data)} months")
            
            # Test each model
            for model_name, model_config in models_config.items():
                current_scenario += 1
                progress = (current_scenario / total_scenarios) * 100
                
                print(f"    🤖 {model_name} ({progress:.0f}% complete)")
                
                try:
                    # Initialize model
                    model_class = model_config['class']
                    
                    if model_name in ['Naive', 'Seasonal_Naive', 'Moving_Average_6m']:
                        # Baseline models
                        model = model_class()
                        
                        if model_name == 'Naive':
                            forecast = model.naive_forecast(train_data, len(test_data))
                        elif model_name == 'Seasonal_Naive':
                            forecast = model.seasonal_naive(train_data, len(test_data))
                        else:  # Moving average
                            forecast = model.moving_average(train_data, len(test_data), 
                                                          window=model_config['params']['window'])
                    
                    else:
                        # Advanced models
                        model_params = model_config.get('params', {})
                        model = model_class(**model_params)
                        
                        # Fit the model
                        if model_config['requires_competitive'] and competitive_features is not None:
                            model.fit(train_data, competitive_features)
                        else:
                            model.fit(train_data, None)
                        
                        # Generate forecast
                        if model_name in ['LSTM']:
                            # LSTM needs special handling
                            forecast = model.predict(
                                periods=len(test_data),
                                competitive_data=competitive_features,
                                last_known_values=train_data
                            )
                        else:
                            # Prophet and XGBoost
                            forecast = model.predict(
                                periods=len(test_data),
                                competitive_data=competitive_features
                            )
                    
                    # Calculate metrics
                    metrics = calculate_metrics(test_data, forecast)
                    
                    # Calculate growth capture
                    if len(test_data) > 1:
                        actual_growth = ((test_data.iloc[-1] / test_data.iloc[0]) - 1) * 100
                        forecast_growth = ((forecast.iloc[-1] / forecast.iloc[0]) - 1) * 100
                        growth_capture = (forecast_growth / actual_growth) * 100 if actual_growth != 0 else 0
                    else:
                        actual_growth = 0
                        forecast_growth = 0
                        growth_capture = 0
                    
                    # Store results
                    result = {
                        'model': model_name,
                        'cutoff_date': cutoff_date,
                        'forecast_horizon': horizon,
                        'train_size': len(train_data),
                        'test_size': len(test_data),
                        'actual_growth_pct': actual_growth,
                        'forecast_growth_pct': forecast_growth,
                        'growth_capture_pct': growth_capture,
                        'has_competitive_features': competitive_features is not None,
                        **metrics
                    }
                    
                    results.append(result)
                    
                    print(f"      ✅ MAPE: {metrics['MAPE']:.1f}%, Growth Capture: {growth_capture:.0f}%")
                    
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
                    
                    # Store error result
                    results.append({
                        'model': model_name,
                        'cutoff_date': cutoff_date,
                        'forecast_horizon': horizon,
                        'error': str(e),
                        'MAPE': np.nan,
                        'RMSE': np.nan,
                        'R2': np.nan,
                        'growth_capture_pct': np.nan,
                        'has_competitive_features': competitive_features is not None
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n✅ Evaluation Complete!")
    print(f"   Generated {len(results_df)} results")
    
    return results_df

def create_comprehensive_visualizations(results_df):
    """Create comprehensive visualizations for all models"""
    
    print("\n📊 Creating comprehensive visualizations...")
    
    # Filter clean results
    clean_results = results_df.dropna(subset=['MAPE'])
    
    if clean_results.empty:
        print("❌ No valid results to visualize")
        return None
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Performance Heatmap
    ax1 = plt.subplot(4, 3, 1)
    
    heatmap_data = clean_results.pivot_table(
        values='MAPE',
        index='cutoff_date',
        columns=['model', 'forecast_horizon'],
        aggfunc='mean'
    )
    
    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   ax=ax1, cbar_kws={'label': 'MAPE (%)'})
        ax1.set_title('MAPE Performance Heatmap\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Model & Horizon')
        ax1.set_ylabel('Cutoff Date')
    
    # 2. Model Performance Comparison
    ax2 = plt.subplot(4, 3, 2)
    
    model_performance = clean_results.groupby('model')['MAPE'].mean().sort_values()
    bars = ax2.bar(range(len(model_performance)), model_performance.values)
    
    # Color bars by performance level
    for i, bar in enumerate(bars):
        value = model_performance.iloc[i]
        if value < 15:
            bar.set_color('green')
        elif value < 25:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax2.set_title('Average MAPE by Model', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_xticks(range(len(model_performance)))
    ax2.set_xticklabels(model_performance.index, rotation=45)
    
    # 3. Growth Capture Analysis
    ax3 = plt.subplot(4, 3, 3)
    
    growth_data = clean_results.groupby('model')['growth_capture_pct'].mean().sort_values(ascending=False)
    bars = ax3.bar(range(len(growth_data)), growth_data.values)
    
    # Color bars based on growth capture
    for i, bar in enumerate(bars):
        value = growth_data.iloc[i]
        if value > 80:
            bar.set_color('green')
        elif value > 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax3.set_title('Growth Trend Capture by Model', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Growth Capture (%)')
    ax3.set_xticks(range(len(growth_data)))
    ax3.set_xticklabels(growth_data.index, rotation=45)
    ax3.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    
    # 4. Temporal Performance
    ax4 = plt.subplot(4, 3, 4)
    
    for model in clean_results['model'].unique():
        model_data = clean_results[clean_results['model'] == model]
        temporal_perf = model_data.groupby('cutoff_date')['MAPE'].mean()
        
        ax4.plot(temporal_perf.index, temporal_perf.values, 
                marker='o', linewidth=2, label=model)
    
    ax4.set_title('MAPE Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Cutoff Date')
    ax4.set_ylabel('MAPE (%)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Horizon Effects
    ax5 = plt.subplot(4, 3, 5)
    
    horizon_data = clean_results.groupby(['model', 'forecast_horizon'])['MAPE'].mean().unstack()
    if not horizon_data.empty:
        horizon_data.plot(kind='bar', ax=ax5, width=0.8)
        ax5.set_title('MAPE by Forecast Horizon', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Model')
        ax5.set_ylabel('MAPE (%)')
        ax5.legend(title='Horizon (months)')
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Model Stability
    ax6 = plt.subplot(4, 3, 6)
    
    stability_data = []
    for model in clean_results['model'].unique():
        model_data = clean_results[clean_results['model'] == model]
        mean_mape = model_data['MAPE'].mean()
        std_mape = model_data['MAPE'].std()
        cv = std_mape / mean_mape if mean_mape > 0 and not pd.isna(std_mape) else 0
        
        stability_data.append({
            'model': model,
            'mean_mape': mean_mape,
            'cv_mape': cv
        })
    
    stability_df = pd.DataFrame(stability_data)
    
    if not stability_df.empty:
        scatter = ax6.scatter(stability_df['mean_mape'], stability_df['cv_mape'], 
                            s=100, alpha=0.7, c=range(len(stability_df)), cmap='viridis')
        
        for i, model in enumerate(stability_df['model']):
            ax6.annotate(model, 
                       (stability_df.iloc[i]['mean_mape'], stability_df.iloc[i]['cv_mape']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax6.set_title('Performance vs Stability\n(Lower-Left is Better)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Mean MAPE (%)')
        ax6.set_ylabel('Coefficient of Variation')
        ax6.grid(True, alpha=0.3)
    
    # 7-12. Individual model comparisons
    models = clean_results['model'].unique()
    
    for i, model in enumerate(models[:6]):  # Show top 6 models
        ax = plt.subplot(4, 3, 7 + i)
        
        model_data = clean_results[clean_results['model'] == model]
        
        # Performance by period and horizon
        model_pivot = model_data.pivot_table(
            values='MAPE',
            index='cutoff_date',
            columns='forecast_horizon',
            aggfunc='mean'
        )
        
        if not model_pivot.empty:
            model_pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{model} - MAPE by Period & Horizon', fontsize=10, fontweight='bold')
            ax.set_xlabel('Cutoff Date')
            ax.set_ylabel('MAPE (%)')
            ax.legend(title='Horizon', fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    
    # Save the comprehensive dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'comprehensive_evaluation_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Saved comprehensive dashboard: {filename}")
    
    plt.show()
    
    return fig

def generate_comprehensive_report(results_df):
    """Generate comprehensive insights report"""
    
    print("\n📋 COMPREHENSIVE EVALUATION REPORT")
    print("=" * 65)
    
    # Filter clean results
    clean_results = results_df.dropna(subset=['MAPE'])
    
    if clean_results.empty:
        print("❌ No valid results to analyze")
        return
    
    # Overall performance ranking
    print("\n🏆 OVERALL PERFORMANCE RANKING:")
    
    model_performance = clean_results.groupby('model').agg({
        'MAPE': ['mean', 'std', 'count'],
        'R2': 'mean',
        'growth_capture_pct': 'mean'
    }).round(2)
    
    model_performance.columns = ['MAPE_mean', 'MAPE_std', 'count', 'R2_mean', 'growth_mean']
    model_performance = model_performance.sort_values('MAPE_mean')
    
    for i, (model, data) in enumerate(model_performance.iterrows()):
        print(f"{i+1}. {model}: MAPE {data['MAPE_mean']:.1f}% ± {data['MAPE_std']:.1f}%, "
              f"R² {data['R2_mean']:.3f}, Growth {data['growth_mean']:.0f}% (n={data['count']})")
    
    # Advanced vs Baseline comparison
    print("\n🤖 ADVANCED vs BASELINE MODELS:")
    
    baseline_models = ['Naive', 'Seasonal_Naive', 'Moving_Average_6m']
    advanced_models = ['Prophet', 'XGBoost', 'LSTM']
    
    baseline_perf = clean_results[clean_results['model'].isin(baseline_models)]['MAPE'].mean()
    advanced_perf = clean_results[clean_results['model'].isin(advanced_models)]['MAPE'].mean()
    
    if not pd.isna(baseline_perf) and not pd.isna(advanced_perf):
        improvement = ((baseline_perf - advanced_perf) / baseline_perf) * 100
        print(f"• Baseline models average: {baseline_perf:.1f}% MAPE")
        print(f"• Advanced models average: {advanced_perf:.1f}% MAPE")
        print(f"• Improvement: {improvement:.1f}%")
    
    # Best performers by category
    print("\n🎯 BEST PERFORMERS BY CATEGORY:")
    
    categories = {
        'Accuracy': ('MAPE', 'min'),
        'Stability': ('MAPE', 'std'),
        'Growth Capture': ('growth_capture_pct', 'max'),
        'Consistency': ('R2', 'max')
    }
    
    for category, (metric, agg_func) in categories.items():
        if agg_func == 'min':
            best_model = clean_results.groupby('model')[metric].mean().idxmin()
            best_value = clean_results.groupby('model')[metric].mean().min()
        elif agg_func == 'max':
            best_model = clean_results.groupby('model')[metric].mean().idxmax()
            best_value = clean_results.groupby('model')[metric].mean().max()
        else:  # std
            best_model = clean_results.groupby('model')[metric].std().idxmin()
            best_value = clean_results.groupby('model')[metric].std().min()
        
        print(f"• {category}: {best_model} ({best_value:.2f})")
    
    # Temporal analysis
    print("\n📅 TEMPORAL PERFORMANCE ANALYSIS:")
    
    for cutoff in clean_results['cutoff_date'].unique():
        period_data = clean_results[clean_results['cutoff_date'] == cutoff]
        best_model = period_data.groupby('model')['MAPE'].mean().idxmin()
        best_mape = period_data.groupby('model')['MAPE'].mean().min()
        worst_model = period_data.groupby('model')['MAPE'].mean().idxmax()
        worst_mape = period_data.groupby('model')['MAPE'].mean().max()
        
        print(f"• {cutoff.strftime('%Y-%m')}: Best = {best_model} ({best_mape:.1f}%), "
              f"Worst = {worst_model} ({worst_mape:.1f}%)")
    
    # Competitive features impact
    print("\n🎯 COMPETITIVE FEATURES IMPACT:")
    
    models_with_competitive = clean_results[clean_results['has_competitive_features'] == True]['model'].unique()
    models_without_competitive = clean_results[clean_results['has_competitive_features'] == False]['model'].unique()
    
    if len(models_with_competitive) > 0 and len(models_without_competitive) > 0:
        with_comp_perf = clean_results[clean_results['has_competitive_features'] == True]['MAPE'].mean()
        without_comp_perf = clean_results[clean_results['has_competitive_features'] == False]['MAPE'].mean()
        
        print(f"• Models with competitive features: {with_comp_perf:.1f}% MAPE")
        print(f"• Models without competitive features: {without_comp_perf:.1f}% MAPE")
        
        impact = ((without_comp_perf - with_comp_perf) / without_comp_perf) * 100
        print(f"• Competitive features impact: {impact:.1f}% improvement")
    
    # Recommendations
    print("\n💡 STRATEGIC RECOMMENDATIONS:")
    
    # Most accurate model
    best_accuracy = model_performance.index[0]
    print(f"• For highest accuracy: Use {best_accuracy}")
    
    # Most stable model
    stability_ranking = clean_results.groupby('model')['MAPE'].std().sort_values()
    most_stable = stability_ranking.index[0]
    print(f"• For stability: Use {most_stable}")
    
    # Best growth capture
    best_growth = clean_results.groupby('model')['growth_capture_pct'].mean().idxmax()
    print(f"• For growth trend capture: Use {best_growth}")
    
    # Production recommendation
    top_performers = model_performance.head(3).index.tolist()
    print(f"• For production deployment: Consider ensemble of {', '.join(top_performers)}")
    
    return clean_results

def main():
    """Main execution function"""
    
    # Run comprehensive evaluation
    results_df = run_comprehensive_evaluation()
    
    if results_df is not None and len(results_df) > 0:
        
        # Create visualizations
        create_comprehensive_visualizations(results_df)
        
        # Generate report
        clean_results = generate_comprehensive_report(results_df)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_filename = f'comprehensive_results_{timestamp}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"\n💾 Detailed results saved: {results_filename}")
        
        # Save clean results
        if clean_results is not None and len(clean_results) > 0:
            clean_filename = f'clean_results_{timestamp}.csv'
            clean_results.to_csv(clean_filename, index=False)
            print(f"💾 Clean results saved: {clean_filename}")
        
        # Display final summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total scenarios: {len(results_df)}")
        print(f"   Successful evaluations: {len(results_df.dropna(subset=['MAPE']))}")
        print(f"   Failed evaluations: {len(results_df[results_df['MAPE'].isna()])}")
        print(f"   Models tested: {results_df['model'].nunique()}")
        print(f"   Advanced models included: Prophet, XGBoost, LSTM")
        print(f"   Baseline models included: Naive, Seasonal_Naive, Moving_Average")
        
        return results_df
    
    else:
        print("❌ No results generated. Check data and configuration.")
        return None

if __name__ == "__main__":
    main() 