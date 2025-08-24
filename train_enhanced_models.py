#!/usr/bin/env python3
"""
Enhanced Models Training Script for MEDIS Sales Forecasting

This script trains and evaluates the enhanced LSTM and Transformer models
designed to better capture the strong growth trends in pharmaceutical sales.

Key Improvements:
- Longer sequence lengths (24 months vs 12)
- Enhanced feature engineering with trend/momentum features
- Log transformation for exponential growth patterns
- RobustScaler for better trend handling
- Advanced architectures (Transformer with attention)

Usage:
    python train_enhanced_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from forecasting_models import (
    load_and_prepare_data, 
    prepare_medis_data,
    LSTMModel,
    EnhancedLSTMModel, 
    TransformerModel,
    ProphetModel,
    ModelEvaluator
)

def main():
    print("🚀 Enhanced MEDIS Sales Forecasting - Training Script")
    print("=" * 60)
    
    # Load and prepare data
    print("\n📂 Loading data...")
    try:
        df = load_and_prepare_data('MEDIS_VENTES.xlsx')
        
        # Filter MEDIS data directly
        medis_df = df[df['laboratoire'] == 'MEDIS'].copy()
        
        print(f"✅ Loaded {len(df):,} total records")
        print(f"✅ MEDIS data: {len(medis_df):,} records")
        
        if len(medis_df) > 0:
            print(f"   Date range: {medis_df['date'].min()} to {medis_df['date'].max()}")
        else:
            print("❌ No MEDIS data found!")
            return
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Prepare time series data
    print("\n📊 Preparing time series...")
    monthly_sales = medis_df.groupby('date')['sales'].sum().fillna(0)
    monthly_sales = monthly_sales.sort_index()
    
    print(f"✅ Monthly time series: {len(monthly_sales)} months")
    print(f"   Latest sales: {monthly_sales.iloc[-1]:,.0f} boxes")
    print(f"   Growth trend: {((monthly_sales.iloc[-1] / monthly_sales.iloc[0]) - 1) * 100:.1f}% total growth")
    
    # Split data for validation (use Nov 2023 as cutoff like in original analysis)
    cutoff_date = '2023-11-30'
    
    train_data = monthly_sales[monthly_sales.index <= cutoff_date]
    test_data = monthly_sales[monthly_sales.index > cutoff_date]
    
    print(f"\n🎯 Training/Validation Split:")
    print(f"   Training: {len(train_data)} months ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"   Validation: {len(test_data)} months ({test_data.index[0]} to {test_data.index[-1]})")
    
    if len(test_data) == 0:
        print("⚠️ No validation data available. Using last 6 months for validation.")
        split_idx = len(monthly_sales) - 6
        train_data = monthly_sales.iloc[:split_idx]
        test_data = monthly_sales.iloc[split_idx:]
    
    # Prepare competitive data
    print("\n🏢 Preparing competitive data...")
    competitive_data = df[df['laboratoire'] != 'MEDIS'].copy()
    print(f"✅ Competitive data: {len(competitive_data):,} records")
    
    # Models to test
    models_to_test = {
        'Original LSTM': LSTMModel(
            sequence_length=12,
            lstm_units=50,
            include_competitive_features=True,
            dropout_rate=0.2
        ),
        'Enhanced LSTM': EnhancedLSTMModel(
            sequence_length=24,
            lstm_units=100,
            include_competitive_features=True,
            dropout_rate=0.2
        ),
        'Transformer': TransformerModel(
            sequence_length=24,
            d_model=64,
            n_heads=4,
            include_competitive_features=True,
            n_layers=2
        ),
        'Prophet (Baseline)': ProphetModel(
            include_competitive_features=True
        )
    }
    
    # Train and evaluate models
    print(f"\n🔬 Training and evaluating {len(models_to_test)} models...")
    
    results = {}
    forecasts = {}
    
    for model_name, model in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"🔄 Training {model_name}...")
        
        try:
            # Fit model
            start_time = datetime.now()
            model.fit(train_data, competitive_data)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Generate forecast
            forecast = model.predict(
                periods=len(test_data),
                competitive_data=competitive_data,
                last_known_values=train_data
            )
            
            forecasts[model_name] = forecast
            
            # Calculate metrics
            evaluator = ModelEvaluator()
            metrics = evaluator.calculate_metrics(test_data, forecast)
            metrics['training_time'] = training_time
            
            results[model_name] = metrics
            
            print(f"✅ {model_name} completed:")
            print(f"   MAPE: {metrics['MAPE']:.1f}%")
            print(f"   RMSE: {metrics['RMSE']:.0f}")
            print(f"   R²: {metrics['R2']:.3f}")
            print(f"   Training time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue
    
    # Display results comparison
    print(f"\n📈 RESULTS COMPARISON")
    print("=" * 60)
    
    if len(results) > 0:
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(3)
        
        print("\nPerformance Metrics:")
        print(results_df[['MAPE', 'RMSE', 'R2', 'Directional_Accuracy']].to_string())
        
        # Find best performers
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Lowest MAPE: {results_df['MAPE'].idxmin()} ({results_df['MAPE'].min():.1f}%)")
        print(f"   Lowest RMSE: {results_df['RMSE'].idxmin()} ({results_df['RMSE'].min():.0f})")
        print(f"   Highest R²: {results_df['R2'].idxmax()} ({results_df['R2'].max():.3f})")
        
        # Create visualization
        print(f"\n📊 Creating visualization...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Time series comparison
        ax1.plot(train_data.index, train_data.values, 
                'b-o', label='Training Data', linewidth=2, markersize=4)
        ax1.plot(test_data.index, test_data.values, 
                'g-o', label='Actual (Validation)', linewidth=3, markersize=5)
        
        colors = ['red', 'orange', 'purple', 'brown']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            ax1.plot(forecast.index, forecast.values,
                    '--s', color=colors[i % len(colors)], 
                    label=f'{model_name}', linewidth=2, markersize=4)
        
        ax1.axvline(x=pd.to_datetime(cutoff_date), color='gray', 
                   linestyle=':', alpha=0.7, linewidth=2, label='Train/Val Split')
        ax1.set_title('Enhanced Models vs Original - Forecasting Performance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales (boxes)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Metrics comparison
        metrics_to_plot = ['MAPE', 'RMSE', 'R2']
        x_pos = np.arange(len(results))
        
        for i, metric in enumerate(metrics_to_plot):
            ax2_sub = plt.subplot(2, 3, 4 + i)
            values = [results[model][metric] for model in results.keys()]
            bars = ax2_sub.bar(range(len(values)), values, color=colors[:len(values)])
            ax2_sub.set_title(f'{metric}')
            ax2_sub.set_xticks(range(len(values)))
            ax2_sub.set_xticklabels([name.replace(' ', '\n') for name in results.keys()], fontsize=8)
            
            # Highlight best performer
            if metric in ['MAPE', 'RMSE']:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            bars[best_idx].set_color('gold')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'enhanced_models_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved as {filename}")
        
        plt.show()
        
        # Analysis and recommendations
        print(f"\n💡 ANALYSIS & RECOMMENDATIONS:")
        print("=" * 60)
        
        # Check if enhanced models are better
        if 'Enhanced LSTM' in results and 'Original LSTM' in results:
            enhanced_mape = results['Enhanced LSTM']['MAPE']
            original_mape = results['Original LSTM']['MAPE']
            improvement = ((original_mape - enhanced_mape) / original_mape) * 100
            
            if improvement > 0:
                print(f"✅ Enhanced LSTM shows {improvement:.1f}% improvement over original LSTM")
            else:
                print(f"⚠️ Enhanced LSTM shows {abs(improvement):.1f}% degradation vs original LSTM")
        
        if 'Transformer' in results:
            transformer_mape = results['Transformer']['MAPE']
            print(f"🤖 Transformer achieved {transformer_mape:.1f}% MAPE")
            
        # Check for strong growth capture
        latest_actual = test_data.iloc[-1]
        for model_name, forecast in forecasts.items():
            latest_forecast = forecast.iloc[-1]
            growth_capture = (latest_forecast / latest_actual) * 100
            print(f"   {model_name}: Capturing {growth_capture:.1f}% of actual growth trend")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   • Actual data shows strong exponential growth")
        print(f"   • Enhanced models use longer sequences (24 vs 12 months)")
        print(f"   • Log transformation helps capture exponential trends")
        print(f"   • RobustScaler better handles outliers and growth")
        print(f"   • Transformer attention mechanism captures long-range dependencies")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Use best performing model in Streamlit dashboard")
        print(f"   2. Consider ensemble of top 2-3 models")
        print(f"   3. Monitor performance on recent data")
        print(f"   4. Retrain monthly with latest data")
        
    else:
        print("❌ No models completed successfully. Check data and dependencies.")

if __name__ == "__main__":
    main() 