#!/usr/bin/env python3
"""
Quick Automated Evaluation Test for MEDIS Models

This demonstrates how to evaluate models across multiple timeframes
and forecast horizons automatically.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from forecasting_models import (
    load_and_prepare_data,
    BaselineModels,
    ProphetModel,
    XGBoostModel,
    ModelEvaluator
)

def run_multi_period_evaluation():
    """
    Run evaluation across multiple cutoff dates and horizons
    """
    print("🚀 Multi-Period Model Evaluation for MEDIS")
    print("=" * 50)
    
    # Load data
    print("\n📂 Loading data...")
    try:
        df = load_and_prepare_data('MEDIS_VENTES.xlsx')
        medis_df = df[df['laboratoire'] == 'MEDIS'].copy()
        competitive_data = df[df['laboratoire'] != 'MEDIS'].copy()
        
        # Create monthly time series
        monthly_sales = medis_df.groupby('date')['sales'].sum().fillna(0).sort_index()
        
        print(f"✅ Data loaded: {len(monthly_sales)} months")
        print(f"   Range: {monthly_sales.index[0]} to {monthly_sales.index[-1]}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Define evaluation parameters
    cutoff_dates = [
        pd.Timestamp('2021-11-01'),
        pd.Timestamp('2022-11-01'), 
        pd.Timestamp('2023-11-01')
    ]
    
    forecast_horizons = [3, 6, 12]  # months
    
    models_to_test = {
        'Prophet': ProphetModel(include_competitive_features=True),
        'XGBoost': XGBoostModel(include_competitive_features=True),
        'Naive': BaselineModels()
    }
    
    print(f"\n🔬 Testing {len(models_to_test)} models across {len(cutoff_dates)} periods")
    print(f"   Cutoff dates: {[d.strftime('%Y-%m') for d in cutoff_dates]}")
    print(f"   Horizons: {forecast_horizons} months")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    results = []
    
    # Run evaluations
    total_scenarios = len(cutoff_dates) * len(forecast_horizons) * len(models_to_test)
    current_scenario = 0
    
    for cutoff_date in cutoff_dates:
        print(f"\n📅 Evaluating cutoff: {cutoff_date.strftime('%Y-%m')}")
        
        # Check if cutoff date exists in data
        if cutoff_date not in monthly_sales.index:
            print(f"   ⚠️  Skipping {cutoff_date} - not in data")
            continue
            
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
            
            # Filter competitive data for training period
            comp_data_filtered = competitive_data[
                competitive_data['date'] <= cutoff_date
            ].copy()
            
            print(f"    📊 Train: {len(train_data)} months, Test: {len(test_data)} months")
            
            # Test each model
            for model_name, model in models_to_test.items():
                current_scenario += 1
                progress = (current_scenario / total_scenarios) * 100
                
                print(f"    🤖 {model_name} ({progress:.0f}% complete)")
                
                try:
                    # Train and predict
                    if model_name == 'Naive':
                        forecast = model.naive_forecast(train_data, len(test_data))
                    else:
                        model.fit(train_data, comp_data_filtered)
                        forecast = model.predict(
                            periods=len(test_data), 
                            competitive_data=comp_data_filtered
                        )
                    
                    # Calculate metrics
                    metrics = evaluator.calculate_metrics(test_data, forecast)
                    
                    # Calculate growth capture
                    actual_growth = ((test_data.iloc[-1] / test_data.iloc[0]) - 1) * 100
                    forecast_growth = ((forecast.iloc[-1] / forecast.iloc[0]) - 1) * 100
                    growth_capture = (forecast_growth / actual_growth) * 100 if actual_growth != 0 else 0
                    
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
                        'growth_capture_pct': np.nan
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n✅ Evaluation Complete!")
    print(f"   Generated {len(results_df)} results")
    
    return results_df

def create_performance_visualizations(results_df):
    """
    Create comprehensive performance visualizations
    """
    print("\n📊 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance Heatmap
    ax1 = axes[0, 0]
    
    # Create pivot table for heatmap
    heatmap_data = results_df.pivot_table(
        values='MAPE',
        index='cutoff_date',
        columns=['model', 'forecast_horizon'],
        aggfunc='mean'
    )
    
    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   ax=ax1, cbar_kws={'label': 'MAPE (%)'})
        ax1.set_title('MAPE Performance Heatmap\n(Lower is Better)')
        ax1.set_xlabel('Model & Horizon')
        ax1.set_ylabel('Cutoff Date')
    
    # 2. Model Performance by Horizon
    ax2 = axes[0, 1]
    
    horizon_data = results_df.groupby(['model', 'forecast_horizon'])['MAPE'].mean().unstack()
    if not horizon_data.empty:
        horizon_data.plot(kind='bar', ax=ax2)
        ax2.set_title('MAPE by Forecast Horizon')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('MAPE (%)')
        ax2.legend(title='Horizon (months)')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Growth Capture Analysis
    ax3 = axes[1, 0]
    
    growth_data = results_df.groupby('model')['growth_capture_pct'].mean().sort_values()
    if not growth_data.empty:
        bars = ax3.bar(range(len(growth_data)), growth_data.values)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            value = growth_data.iloc[i]
            if value > 90:
                bar.set_color('green')
            elif value > 70:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax3.set_title('Growth Trend Capture\n(% of Actual Growth)')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Growth Capture (%)')
        ax3.set_xticks(range(len(growth_data)))
        ax3.set_xticklabels(growth_data.index, rotation=45)
        ax3.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    
    # 4. Temporal Stability
    ax4 = axes[1, 1]
    
    # Calculate stability metrics
    stability_data = []
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        mean_mape = model_data['MAPE'].mean()
        std_mape = model_data['MAPE'].std()
        cv = std_mape / mean_mape if mean_mape > 0 and not pd.isna(std_mape) else np.inf
        
        stability_data.append({
            'model': model,
            'mean_mape': mean_mape,
            'cv_mape': cv
        })
    
    stability_df = pd.DataFrame(stability_data)
    
    if not stability_df.empty and not stability_df['mean_mape'].isna().all():
        scatter = ax4.scatter(stability_df['mean_mape'], stability_df['cv_mape'], 
                            s=100, alpha=0.7)
        
        for i, model in enumerate(stability_df['model']):
            if not pd.isna(stability_df.iloc[i]['mean_mape']):
                ax4.annotate(model, 
                           (stability_df.iloc[i]['mean_mape'], stability_df.iloc[i]['cv_mape']),
                           xytext=(5, 5), textcoords='offset points')
        
        ax4.set_title('Performance vs Stability\n(Lower-Left is Better)')
        ax4.set_xlabel('Mean MAPE (%)')
        ax4.set_ylabel('Coefficient of Variation')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'multi_period_evaluation_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Saved visualization: {filename}")
    
    plt.show()
    
    return fig

def generate_summary_report(results_df):
    """
    Generate summary report with key insights
    """
    print("\n📋 PERFORMANCE SUMMARY REPORT")
    print("=" * 50)
    
    # Overall performance
    print("\n🏆 BEST PERFORMERS:")
    
    # Best MAPE
    best_mape_model = results_df.groupby('model')['MAPE'].mean().idxmin()
    best_mape_value = results_df.groupby('model')['MAPE'].mean().min()
    print(f"• Lowest MAPE: {best_mape_model} ({best_mape_value:.1f}%)")
    
    # Best growth capture
    best_growth_model = results_df.groupby('model')['growth_capture_pct'].mean().idxmax()
    best_growth_value = results_df.groupby('model')['growth_capture_pct'].mean().max()
    print(f"• Best Growth Capture: {best_growth_model} ({best_growth_value:.0f}%)")
    
    # Temporal stability
    print("\n⚖️ TEMPORAL STABILITY:")
    
    stability_scores = {}
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        if not model_data['MAPE'].isna().all():
            cv = model_data['MAPE'].std() / model_data['MAPE'].mean()
            stability_scores[model] = 1 - cv if not pd.isna(cv) else 0
    
    # Sort by stability
    sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(sorted_stability[:3]):
        print(f"{i+1}. {model}: {score:.3f}")
    
    # Horizon effects
    print("\n⏱️ HORIZON EFFECTS:")
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        horizon_perf = model_data.groupby('forecast_horizon')['MAPE'].mean()
        
        if len(horizon_perf) > 1:
            best_horizon = horizon_perf.idxmin()
            worst_horizon = horizon_perf.idxmax()
            degradation = horizon_perf.max() - horizon_perf.min()
            
            print(f"• {model}: Best at {best_horizon}m, degrades {degradation:.1f}% by {worst_horizon}m")
    
    # Period-specific insights
    print("\n📅 PERIOD-SPECIFIC INSIGHTS:")
    
    for cutoff in results_df['cutoff_date'].unique():
        period_data = results_df[results_df['cutoff_date'] == cutoff]
        best_period_model = period_data.groupby('model')['MAPE'].mean().idxmin()
        best_period_mape = period_data.groupby('model')['MAPE'].mean().min()
        
        print(f"• {cutoff.strftime('%Y-%m')}: Best = {best_period_model} ({best_period_mape:.1f}%)")
    
    # Key recommendations
    print("\n💡 KEY RECOMMENDATIONS:")
    
    # Find most consistent performer
    consistent_models = []
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        success_rate = (1 - model_data['MAPE'].isna().sum() / len(model_data)) * 100
        
        if success_rate >= 80:  # At least 80% success rate
            mean_mape = model_data['MAPE'].mean()
            if mean_mape <= 25:  # Reasonable accuracy
                consistent_models.append((model, mean_mape, success_rate))
    
    if consistent_models:
        best_consistent = min(consistent_models, key=lambda x: x[1])
        print(f"• Most reliable model: {best_consistent[0]} (MAPE: {best_consistent[1]:.1f}%, Success: {best_consistent[2]:.0f}%)")
    
    # Identify problematic periods
    problematic_periods = []
    for cutoff in results_df['cutoff_date'].unique():
        period_data = results_df[results_df['cutoff_date'] == cutoff]
        avg_mape = period_data['MAPE'].mean()
        
        if avg_mape > 30:  # High error period
            problematic_periods.append((cutoff, avg_mape))
    
    if problematic_periods:
        print("• Challenging periods detected:")
        for period, mape in problematic_periods:
            print(f"  - {period.strftime('%Y-%m')}: Avg MAPE {mape:.1f}%")
    
    return results_df

def main():
    """Main execution function"""
    
    # Run evaluation
    results_df = run_multi_period_evaluation()
    
    if results_df is not None and len(results_df) > 0:
        # Create visualizations
        create_performance_visualizations(results_df)
        
        # Generate report
        generate_summary_report(results_df)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation_results_{timestamp}.csv'
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
        
        # Display sample results
        print(f"\n📊 Sample Results:")
        print(results_df[['model', 'cutoff_date', 'forecast_horizon', 'MAPE', 'growth_capture_pct']].head(10))
    
    else:
        print("❌ No results generated. Check data and model configuration.")

if __name__ == "__main__":
    main() 