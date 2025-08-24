#!/usr/bin/env python3
"""
MEDIS Sales Forecasting - Model Training Script
Train all models on data until November 2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, date
import pickle
import os

# Import our forecasting models
from forecasting_models import (
    BaselineModels, ProphetModel, XGBoostModel, LSTMModel, 
    EnsembleModel, ModelEvaluator, load_and_prepare_data
)

def main():
    """Main training pipeline"""
    
    print("🚀 MEDIS Sales Forecasting - Model Training Pipeline")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n📊 1. Loading and preparing data...")
    try:
        df = load_and_prepare_data('MEDIS_VENTES.xlsx')
        print(f"✅ Loaded {len(df):,} records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Filter for MEDIS data
        medis_data = df[df['laboratoire'] == 'MEDIS'].copy()
        print(f"✅ MEDIS data: {len(medis_data):,} records")
        
        # Aggregate sales by date
        sales_ts = medis_data.groupby('date')['VENTE_IMS'].sum().fillna(0)
        sales_ts.index = pd.to_datetime(sales_ts.index)
        sales_ts = sales_ts.sort_index()
        
        print(f"✅ Time series prepared: {len(sales_ts)} monthly data points")
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # 2. Split data at November 2023
    print("\n📅 2. Splitting data at November 2023...")
    cutoff_date = pd.to_datetime('2023-11-30')
    
    train_ts = sales_ts[sales_ts.index <= cutoff_date]
    test_ts = sales_ts[sales_ts.index > cutoff_date]
    
    print(f"✅ Training data: {len(train_ts)} points ({train_ts.index.min()} to {train_ts.index.max()})")
    print(f"✅ Test data: {len(test_ts)} points ({test_ts.index.min()} to {test_ts.index.max()})")
    print(f"   Training period: {(train_ts.index.max() - train_ts.index.min()).days / 365.25:.1f} years")
    
    # 3. Prepare competitive data
    print("\n🏢 3. Preparing competitive intelligence...")
    try:
        # Get all non-MEDIS data for training period
        # Convert cutoff_date to match the format in the dataset
        cutoff_str = cutoff_date.strftime('%Y-%m-01')
        
        competitive_data = df[
            (df['laboratoire'] != 'MEDIS') & 
            (df['date'] <= cutoff_date)
        ].copy()
        
        print(f"✅ Competitive data: {len(competitive_data):,} records from {competitive_data['laboratoire'].nunique()} competitors")
        print(f"   Competitors: {', '.join(competitive_data['laboratoire'].unique()[:5])}...")
        
    except Exception as e:
        print(f"⚠️ Could not prepare competitive data: {str(e)}")
        competitive_data = None
    
    # 4. Initialize models
    print("\n🤖 4. Initializing models...")
    
    models = {
        'baseline': BaselineModels(),
        'prophet': ProphetModel(include_competitive_features=True),
        'xgboost': XGBoostModel(include_competitive_features=True, max_lags=6),
        'lstm': LSTMModel(
            sequence_length=min(12, len(train_ts) // 3), 
            lstm_units=64,
            include_competitive_features=True,
            dropout_rate=0.3
        )
    }
    
    print(f"✅ Initialized {len(models)} model types")
    
    # 5. Train models
    print("\n🔄 5. Training models...")
    trained_models = {}
    forecasts = {}
    
    # Train Prophet
    print("\n📈 Training Prophet model...")
    try:
        prophet_model = models['prophet']
        prophet_model.fit(train_ts, competitive_data)
        forecast = prophet_model.predict(len(test_ts), competitive_data)
        
        trained_models['Prophet'] = prophet_model
        forecasts['Prophet'] = forecast
        print(f"✅ Prophet training completed - forecast: {len(forecast)} periods")
        
    except Exception as e:
        print(f"❌ Prophet training failed: {str(e)}")
    
    # Train XGBoost
    print("\n🌲 Training XGBoost model...")
    try:
        xgb_model = models['xgboost']
        xgb_model.fit(train_ts, competitive_data)
        forecast = xgb_model.predict(
            periods=len(test_ts),
            competitive_data=competitive_data,
            last_known_values=train_ts
        )
        
        trained_models['XGBoost'] = xgb_model
        forecasts['XGBoost'] = forecast
        print(f"✅ XGBoost training completed - forecast: {len(forecast)} periods")
        
    except Exception as e:
        print(f"❌ XGBoost training failed: {str(e)}")
    
    # Train LSTM
    print("\n🧠 Training LSTM model...")
    try:
        lstm_model = models['lstm']
        print(f"   LSTM config: {lstm_model.sequence_length} sequence length, {lstm_model.lstm_units} units")
        
        lstm_model.fit(train_ts, competitive_data)
        forecast = lstm_model.predict(
            periods=len(test_ts),
            competitive_data=competitive_data,
            last_known_values=train_ts
        )
        
        trained_models['LSTM'] = lstm_model
        forecasts['LSTM'] = forecast
        print(f"✅ LSTM training completed - forecast: {len(forecast)} periods")
        
        # Plot LSTM training history
        lstm_model.plot_training_history()
        
    except Exception as e:
        print(f"❌ LSTM training failed: {str(e)}")
    
    # Generate baseline forecasts
    print("\n📊 Generating baseline forecasts...")
    try:
        baseline = models['baseline']
        
        # Simple baselines
        baseline_forecasts = {
            'Naive': baseline.naive_forecast(train_ts, len(test_ts)),
            'Seasonal_Naive': baseline.seasonal_naive(train_ts, len(test_ts)),
            'Moving_Average': baseline.moving_average(train_ts, len(test_ts), window=12)
        }
        
        for name, forecast in baseline_forecasts.items():
            forecasts[name] = forecast
            trained_models[name] = baseline
            
        print(f"✅ Generated {len(baseline_forecasts)} baseline forecasts")
        
    except Exception as e:
        print(f"❌ Baseline forecasting failed: {str(e)}")
    
    # Create ensemble
    print("\n🎯 Creating ensemble model...")
    try:
        if len(trained_models) >= 2:
            ensemble = EnsembleModel(ensemble_method='weighted_average')
            
            # Define weights based on model sophistication
            model_weights = {
                'LSTM': 2.5,
                'Prophet': 2.0,
                'XGBoost': 2.0,
                'Moving_Average': 1.0,
                'Seasonal_Naive': 0.6,
                'Naive': 0.4
            }
            
            # Add models to ensemble
            for name, model in trained_models.items():
                weight = model_weights.get(name, 1.0)
                ensemble.add_model(name, model, weight)
            
            # Generate ensemble forecast
            ensemble_forecast = ensemble.predict_ensemble(
                periods=len(test_ts),
                competitive_data=competitive_data,
                last_known_values=train_ts
            )
            
            forecasts['Ensemble'] = ensemble_forecast
            trained_models['Ensemble'] = ensemble
            
            print(f"✅ Ensemble model created with {len(trained_models)-1} base models")
            
            # Show model contributions
            contributions = ensemble.get_model_contributions()
            print("\n📊 Ensemble Model Weights:")
            for model, weight in contributions['contribution'].items():
                print(f"   {model}: {weight:.3f}")
                
        else:
            print("⚠️ Not enough models for ensemble")
            
    except Exception as e:
        print(f"❌ Ensemble creation failed: {str(e)}")
    
    # 6. Evaluate models
    print("\n📏 6. Evaluating model performance...")
    
    evaluator = ModelEvaluator()
    results = {}
    
    for model_name, forecast in forecasts.items():
        try:
            metrics = evaluator.calculate_metrics(test_ts, forecast)
            results[model_name] = metrics
            
            mape = metrics.get('MAPE', np.inf)
            rmse = metrics.get('RMSE', np.inf)
            r2 = metrics.get('R2', 0)
            
            print(f"✅ {model_name:15} - MAPE: {mape:6.2f}%, RMSE: {rmse:8.0f}, R²: {r2:6.3f}")
            
        except Exception as e:
            print(f"❌ {model_name:15} - Evaluation failed: {str(e)}")
    
    # 7. Visualize results
    print("\n📊 7. Creating visualizations...")
    
    # Main comparison chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot complete actual data
    full_ts = pd.concat([train_ts, test_ts])
    ax.plot(full_ts.index, full_ts.values, 
           color='blue', linewidth=3, marker='o', markersize=4,
           label='Actual Sales (Ground Truth)', alpha=0.8)
    
    # Highlight validation period
    ax.axvspan(test_ts.index.min(), test_ts.index.max(), 
              alpha=0.2, color='lightgreen', label='Validation Period')
    
    # Plot forecasts
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'lime', 'gold']
    
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        ax.plot(forecast.index, forecast.values,
               color=colors[i % len(colors)], linewidth=3, 
               marker='s', markersize=5, linestyle='--',
               label=f'{model_name} Forecast')
    
    # Add cutoff line
    ax.axvline(x=cutoff_date, color='gray', linestyle=':', alpha=0.7, linewidth=3, 
              label='Training Cutoff (Nov 2023)')
    
    # Formatting
    ax.set_title('MEDIS Sales Forecasting - Model Comparison\nTraining: Apr 2018 - Nov 2023', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales (boxes)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('medis_forecast_comparison_nov2023.png', dpi=300, bbox_inches='tight')
    print("✅ Saved comparison chart: medis_forecast_comparison_nov2023.png")
    plt.show()
    
    # Performance metrics table
    if results:
        results_df = pd.DataFrame(results).T.round(3)
        
        print("\n📈 Performance Metrics Summary:")
        print("=" * 80)
        print(results_df.to_string())
        
        # Save results
        results_df.to_csv('model_performance_nov2023.csv')
        print("✅ Saved performance metrics: model_performance_nov2023.csv")
        
        # Find best model
        if 'MAPE' in results_df.columns:
            best_model = results_df['MAPE'].idxmin()
            best_mape = results_df.loc[best_model, 'MAPE']
            
            print(f"\n🏆 Best performing model: {best_model}")
            print(f"   MAPE: {best_mape:.2f}%")
            print(f"   RMSE: {results_df.loc[best_model, 'RMSE']:.0f}")
            print(f"   R²: {results_df.loc[best_model, 'R2']:.3f}")
    
    # 8. Save trained models
    print("\n💾 8. Saving trained models...")
    
    models_dir = 'trained_models_nov2023'
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model in trained_models.items():
        try:
            # Save with pickle
            model_path = os.path.join(models_dir, f'{model_name.lower()}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Saved {model_name} model: {model_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save {model_name}: {str(e)}")
    
    # Save training metadata
    metadata = {
        'training_cutoff': cutoff_date.strftime('%Y-%m-%d'),
        'training_records': len(train_ts),
        'test_records': len(test_ts),
        'models_trained': list(trained_models.keys()),
        'best_model': best_model if 'best_model' in locals() else None,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    with open(os.path.join(models_dir, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved training metadata")
    
    # 9. Summary
    print("\n🎉 Training Pipeline Completed!")
    print("=" * 60)
    print(f"📊 Models trained: {len(trained_models)}")
    print(f"📈 Performance evaluated on {len(test_ts)} test periods")
    print(f"💾 Models saved to: {models_dir}/")
    print(f"📋 Results saved to: model_performance_nov2023.csv")
    print(f"📊 Chart saved to: medis_forecast_comparison_nov2023.png")
    
    if 'best_model' in locals():
        print(f"\n🏆 Recommended model: {best_model} (MAPE: {best_mape:.2f}%)")
    
    print("\n✨ Ready for production forecasting!")

if __name__ == "__main__":
    main() 