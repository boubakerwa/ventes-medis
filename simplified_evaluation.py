#!/usr/bin/env python3
"""
Simplified Automated Model Evaluation for MEDIS

This script demonstrates automated evaluation across multiple timeframes
and forecast horizons with working implementations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from forecasting_models import load_and_prepare_data, BaselineModels

def calculate_metrics(actual, forecast):
    """Calculate evaluation metrics"""
    
    # Convert to numpy arrays
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    actual = actual[mask]
    forecast = forecast[mask]
    
    if len(actual) == 0:
        return {
            'MAPE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Directional_Accuracy': np.nan
        }
    
    # MAPE
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    # RMSE
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    
    # MAE
    mae = np.mean(np.abs(actual - forecast))
    
    # R²
    ss_res = np.sum((actual - forecast) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Directional accuracy
    if len(actual) > 1:
        actual_direction = np.diff(actual) > 0
        forecast_direction = np.diff(forecast) > 0
        directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
    else:
        directional_accuracy = np.nan
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

def run_simplified_evaluation():
    """Run simplified evaluation across multiple periods"""
    
    print("🚀 Simplified Multi-Period Model Evaluation")
    print("=" * 55)
    
    # Load data
    print("\n📂 Loading data...")
    try:
        df = load_and_prepare_data('MEDIS_VENTES.xlsx')
        medis_df = df[df['laboratoire'] == 'MEDIS'].copy()
        
        # Create monthly time series
        monthly_sales = medis_df.groupby('date')['sales'].sum().fillna(0).sort_index()
        
        print(f"✅ Data loaded: {len(monthly_sales)} months")
        print(f"   Range: {monthly_sales.index[0]} to {monthly_sales.index[-1]}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Define evaluation parameters
    cutoff_dates = [
        pd.Timestamp('2021-11-01'),
        pd.Timestamp('2022-11-01'), 
        pd.Timestamp('2023-11-01')
    ]
    
    forecast_horizons = [3, 6, 12]  # months
    
    print(f"\n🔬 Testing baseline models across {len(cutoff_dates)} periods")
    print(f"   Cutoff dates: {[d.strftime('%Y-%m') for d in cutoff_dates]}")
    print(f"   Horizons: {forecast_horizons} months")
    
    # Initialize models
    baseline_models = BaselineModels()
    
    results = []
    
    # Run evaluations
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
            
            print(f"    📊 Train: {len(train_data)} months, Test: {len(test_data)} months")
            
            # Test baseline models
            models_to_test = {
                'Naive': lambda: baseline_models.naive_forecast(train_data, len(test_data)),
                'Seasonal_Naive': lambda: baseline_models.seasonal_naive(train_data, len(test_data)),
                'Moving_Average_6m': lambda: baseline_models.moving_average(train_data, len(test_data), window=6),
                'Moving_Average_12m': lambda: baseline_models.moving_average(train_data, len(test_data), window=12),
                'Exponential_Smoothing': lambda: baseline_models.exponential_smoothing(train_data, len(test_data))
            }
            
            for model_name, model_func in models_to_test.items():
                print(f"    🤖 {model_name}")
                
                try:
                    # Generate forecast
                    forecast = model_func()
                    
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

def create_evaluation_dashboard(results_df):
    """Create comprehensive evaluation dashboard"""
    
    print("\n📊 Creating evaluation dashboard...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Heatmap
    ax1 = plt.subplot(3, 3, 1)
    
    # Filter out error results for heatmap
    clean_results = results_df.dropna(subset=['MAPE'])
    
    if not clean_results.empty:
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
    
    # 2. Performance by Horizon
    ax2 = plt.subplot(3, 3, 2)
    
    if not clean_results.empty:
        horizon_data = clean_results.groupby(['model', 'forecast_horizon'])['MAPE'].mean().unstack()
        
        if not horizon_data.empty:
            horizon_data.plot(kind='bar', ax=ax2, width=0.8)
            ax2.set_title('MAPE by Forecast Horizon', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('MAPE (%)')
            ax2.legend(title='Horizon (months)', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)
    
    # 3. Growth Capture Analysis
    ax3 = plt.subplot(3, 3, 3)
    
    if not clean_results.empty:
        growth_data = clean_results.groupby('model')['growth_capture_pct'].mean().sort_values()
        
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
            
            ax3.set_title('Growth Trend Capture\n(% of Actual Growth)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Growth Capture (%)')
            ax3.set_xticks(range(len(growth_data)))
            ax3.set_xticklabels(growth_data.index, rotation=45)
            ax3.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    
    # 4. Temporal Performance (MAPE over time)
    ax4 = plt.subplot(3, 3, 4)
    
    if not clean_results.empty:
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
    
    # 5. Model Stability (CV vs Mean MAPE)
    ax5 = plt.subplot(3, 3, 5)
    
    if not clean_results.empty:
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
            scatter = ax5.scatter(stability_df['mean_mape'], stability_df['cv_mape'], 
                                s=100, alpha=0.7, c=range(len(stability_df)), cmap='viridis')
            
            for i, model in enumerate(stability_df['model']):
                ax5.annotate(model, 
                           (stability_df.iloc[i]['mean_mape'], stability_df.iloc[i]['cv_mape']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax5.set_title('Performance vs Stability\n(Lower-Left is Better)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Mean MAPE (%)')
            ax5.set_ylabel('Coefficient of Variation')
            ax5.grid(True, alpha=0.3)
    
    # 6-9. Individual model performance over periods and horizons
    models = clean_results['model'].unique() if not clean_results.empty else []
    
    for i, model in enumerate(models[:4]):  # Show first 4 models
        ax = plt.subplot(3, 3, 6 + i)
        
        model_data = clean_results[clean_results['model'] == model]
        
        # Performance by horizon for this model
        model_horizon_data = model_data.groupby(['cutoff_date', 'forecast_horizon'])['MAPE'].mean().unstack()
        
        if not model_horizon_data.empty:
            model_horizon_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{model} - MAPE by Period & Horizon', fontsize=10, fontweight='bold')
            ax.set_xlabel('Cutoff Date')
            ax.set_ylabel('MAPE (%)')
            ax.legend(title='Horizon', fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    
    # Save the dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'evaluation_dashboard_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Saved dashboard: {filename}")
    
    plt.show()
    
    return fig

def generate_insights_report(results_df):
    """Generate detailed insights report"""
    
    print("\n📋 DETAILED EVALUATION INSIGHTS")
    print("=" * 55)
    
    # Filter clean results
    clean_results = results_df.dropna(subset=['MAPE'])
    
    if clean_results.empty:
        print("❌ No valid results to analyze")
        return
    
    # Overall performance ranking
    print("\n🏆 OVERALL PERFORMANCE RANKING (by MAPE):")
    model_performance = clean_results.groupby('model')['MAPE'].agg(['mean', 'std', 'count']).round(2)
    model_performance = model_performance.sort_values('mean')
    
    for i, (model, data) in enumerate(model_performance.iterrows()):
        print(f"{i+1}. {model}: {data['mean']:.1f}% ± {data['std']:.1f}% (n={data['count']})")
    
    # Best performers by period
    print("\n📅 BEST PERFORMERS BY PERIOD:")
    
    for cutoff in clean_results['cutoff_date'].unique():
        period_data = clean_results[clean_results['cutoff_date'] == cutoff]
        best_model = period_data.groupby('model')['MAPE'].mean().idxmin()
        best_mape = period_data.groupby('model')['MAPE'].mean().min()
        
        print(f"• {cutoff.strftime('%Y-%m')}: {best_model} ({best_mape:.1f}%)")
    
    # Best performers by horizon
    print("\n⏱️ BEST PERFORMERS BY HORIZON:")
    
    for horizon in clean_results['forecast_horizon'].unique():
        horizon_data = clean_results[clean_results['forecast_horizon'] == horizon]
        best_model = horizon_data.groupby('model')['MAPE'].mean().idxmin()
        best_mape = horizon_data.groupby('model')['MAPE'].mean().min()
        
        print(f"• {horizon}m horizon: {best_model} ({best_mape:.1f}%)")
    
    # Temporal stability analysis
    print("\n⚖️ TEMPORAL STABILITY ANALYSIS:")
    
    stability_scores = {}
    for model in clean_results['model'].unique():
        model_data = clean_results[clean_results['model'] == model]
        cv = model_data['MAPE'].std() / model_data['MAPE'].mean()
        stability_scores[model] = 1 - cv  # Higher is more stable
    
    sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(sorted_stability):
        print(f"{i+1}. {model}: {score:.3f} stability score")
    
    # Growth capture analysis
    print("\n📈 GROWTH CAPTURE ANALYSIS:")
    
    growth_performance = clean_results.groupby('model')['growth_capture_pct'].agg(['mean', 'std']).round(1)
    growth_performance = growth_performance.sort_values('mean', ascending=False)
    
    for model, data in growth_performance.iterrows():
        print(f"• {model}: {data['mean']:.0f}% ± {data['std']:.0f}%")
    
    # Identify best model for different scenarios
    print("\n💡 SCENARIO-BASED RECOMMENDATIONS:")
    
    # Most accurate overall
    best_overall = model_performance.index[0]
    print(f"• Most accurate overall: {best_overall}")
    
    # Most stable
    most_stable = sorted_stability[0][0]
    print(f"• Most stable: {most_stable}")
    
    # Best growth capture
    best_growth = growth_performance.index[0]
    print(f"• Best growth capture: {best_growth}")
    
    # Best for short-term (3m)
    short_term = clean_results[clean_results['forecast_horizon'] == 3]
    if not short_term.empty:
        best_short = short_term.groupby('model')['MAPE'].mean().idxmin()
        print(f"• Best for short-term (3m): {best_short}")
    
    # Best for long-term (12m)
    long_term = clean_results[clean_results['forecast_horizon'] == 12]
    if not long_term.empty:
        best_long = long_term.groupby('model')['MAPE'].mean().idxmin()
        print(f"• Best for long-term (12m): {best_long}")
    
    # Performance degradation analysis
    print("\n📉 HORIZON DEGRADATION ANALYSIS:")
    
    for model in clean_results['model'].unique():
        model_data = clean_results[clean_results['model'] == model]
        horizon_perf = model_data.groupby('forecast_horizon')['MAPE'].mean()
        
        if len(horizon_perf) > 1:
            best_horizon = horizon_perf.idxmin()
            worst_horizon = horizon_perf.idxmax()
            degradation = horizon_perf.max() - horizon_perf.min()
            
            print(f"• {model}: Best at {best_horizon}m ({horizon_perf.min():.1f}%), "
                  f"degrades {degradation:.1f}% by {worst_horizon}m")

def main():
    """Main execution function"""
    
    # Run evaluation
    results_df = run_simplified_evaluation()
    
    if results_df is not None and len(results_df) > 0:
        
        # Create dashboard
        create_evaluation_dashboard(results_df)
        
        # Generate insights
        generate_insights_report(results_df)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation_results_{timestamp}.csv'
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
        
        # Display summary
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Total scenarios tested: {len(results_df)}")
        print(f"   Successful evaluations: {len(results_df.dropna(subset=['MAPE']))}")
        print(f"   Models tested: {results_df['model'].nunique()}")
        print(f"   Time periods: {results_df['cutoff_date'].nunique()}")
        print(f"   Forecast horizons: {sorted(results_df['forecast_horizon'].unique())}")
        
        return results_df
    
    else:
        print("❌ No results generated. Check data and configuration.")
        return None

if __name__ == "__main__":
    main() 