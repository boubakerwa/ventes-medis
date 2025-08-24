#!/usr/bin/env python3
"""
Automated Model Evaluation Framework for MEDIS Sales Forecasting

This script performs comprehensive walk-forward validation across:
- Multiple cutoff dates (temporal robustness)
- Multiple forecast horizons (3, 6, 12 months)
- All available models (Enhanced LSTM, Transformer, Prophet, etc.)

Features:
- Automated performance matrix generation
- Statistical significance testing
- Interactive visualizations
- Detailed reporting with insights

Usage:
    python automated_model_evaluation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from forecasting_models import (
    load_and_prepare_data,
    BaselineModels,
    ProphetModel,
    XGBoostModel,
    LSTMModel,
    EnhancedLSTMModel,
    TransformerModel,
    EnsembleModel,
    ModelEvaluator
)

class WalkForwardValidator:
    """
    Comprehensive walk-forward validation framework
    """
    
    def __init__(self, min_train_months=36, max_test_months=12):
        """
        Initialize validator
        
        Args:
            min_train_months: Minimum training period (months)
            max_test_months: Maximum forecast horizon (months)
        """
        self.min_train_months = min_train_months
        self.max_test_months = max_test_months
        self.results = []
        self.models_cache = {}
        self.evaluator = ModelEvaluator()
        
    def generate_validation_schedule(self, data_series, forecast_horizons=[3, 6, 12]):
        """
        Generate all combinations of cutoff dates and forecast horizons
        
        Args:
            data_series: Time series data
            forecast_horizons: List of forecast horizons to test
            
        Returns:
            List of validation configurations
        """
        schedule = []
        
        # Generate cutoff dates (every 6 months)
        start_date = data_series.index[self.min_train_months]
        end_date = data_series.index[-self.max_test_months - 1]
        
        current_date = start_date
        while current_date <= end_date:
            for horizon in forecast_horizons:
                # Check if we have enough data for this horizon
                cutoff_idx = data_series.index.get_loc(current_date)
                
                if cutoff_idx + horizon < len(data_series):
                    train_data = data_series[:cutoff_idx + 1]
                    test_data = data_series[cutoff_idx + 1:cutoff_idx + 1 + horizon]
                    
                    if len(train_data) >= self.min_train_months and len(test_data) == horizon:
                        schedule.append({
                            'cutoff_date': current_date,
                            'forecast_horizon': horizon,
                            'train_start': train_data.index[0],
                            'train_end': train_data.index[-1],
                            'test_start': test_data.index[0],
                            'test_end': test_data.index[-1],
                            'train_size': len(train_data),
                            'test_size': len(test_data)
                        })
            
            # Move to next cutoff (every 6 months)
            current_date = current_date + pd.DateOffset(months=6)
        
        return schedule
    
    def initialize_models(self):
        """Initialize all models for testing"""
        
        models = {
            'Naive': BaselineModels(),
            'Seasonal_Naive': BaselineModels(),
            'Moving_Average_6m': BaselineModels(),
            'Prophet': ProphetModel(include_competitive_features=True),
            'XGBoost': XGBoostModel(include_competitive_features=True, max_lags=6),
            'LSTM': LSTMModel(
                sequence_length=12,
                lstm_units=50,
                include_competitive_features=True
            ),
            'Enhanced_LSTM': EnhancedLSTMModel(
                sequence_length=24,
                lstm_units=100,
                include_competitive_features=True
            ),
            'Transformer': TransformerModel(
                sequence_length=24,
                d_model=64,
                n_heads=4,
                include_competitive_features=True
            )
        }
        
        return models
    
    def train_and_evaluate_model(self, model_name, model, train_data, test_data, competitive_data, config):
        """
        Train and evaluate a single model
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            print(f"  📊 {model_name} (Horizon: {config['forecast_horizon']}m)...")
            
            start_time = datetime.now()
            
            # Handle baseline models
            if model_name in ['Naive', 'Seasonal_Naive', 'Moving_Average_6m']:
                if model_name == 'Naive':
                    forecast = model.naive_forecast(train_data, len(test_data))
                elif model_name == 'Seasonal_Naive':
                    forecast = model.seasonal_naive(train_data, len(test_data))
                else:
                    forecast = model.moving_average(train_data, len(test_data), window=6)
            
            else:
                # Advanced models
                model.fit(train_data, competitive_data)
                
                if model_name in ['Enhanced_LSTM', 'Transformer', 'LSTM']:
                    forecast = model.predict(
                        periods=len(test_data),
                        competitive_data=competitive_data,
                        last_known_values=train_data
                    )
                else:
                    forecast = model.predict(
                        periods=len(test_data),
                        competitive_data=competitive_data
                    )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(test_data, forecast)
            
            # Add additional metadata
            result = {
                'model': model_name,
                'cutoff_date': config['cutoff_date'],
                'forecast_horizon': config['forecast_horizon'],
                'train_start': config['train_start'],
                'train_end': config['train_end'],
                'test_start': config['test_start'],
                'test_end': config['test_end'],
                'train_size': config['train_size'],
                'test_size': config['test_size'],
                'training_time': training_time,
                **metrics
            }
            
            # Add trend analysis
            actual_growth = ((test_data.iloc[-1] / test_data.iloc[0]) - 1) * 100
            forecast_growth = ((forecast.iloc[-1] / forecast.iloc[0]) - 1) * 100
            growth_capture = (forecast_growth / actual_growth) * 100 if actual_growth != 0 else 0
            
            result['actual_growth_pct'] = actual_growth
            result['forecast_growth_pct'] = forecast_growth
            result['growth_capture_pct'] = growth_capture
            
            return result
            
        except Exception as e:
            print(f"    ❌ Error with {model_name}: {str(e)}")
            return {
                'model': model_name,
                'cutoff_date': config['cutoff_date'],
                'forecast_horizon': config['forecast_horizon'],
                'error': str(e),
                **{metric: np.nan for metric in ['MAPE', 'RMSE', 'R2', 'MAE', 'Directional_Accuracy']}
            }
    
    def run_comprehensive_evaluation(self, data_series, competitive_data, models_to_test=None):
        """
        Run comprehensive walk-forward validation
        
        Args:
            data_series: Time series data
            competitive_data: Competitive intelligence data
            models_to_test: List of model names to test (None = all)
            
        Returns:
            DataFrame with all results
        """
        print("🚀 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        # Generate validation schedule
        schedule = self.generate_validation_schedule(data_series)
        print(f"📅 Generated {len(schedule)} validation scenarios")
        
        # Initialize models
        models = self.initialize_models()
        if models_to_test:
            models = {k: v for k, v in models.items() if k in models_to_test}
        
        print(f"🔬 Testing {len(models)} models: {list(models.keys())}")
        
        # Run evaluations
        all_results = []
        
        for i, config in enumerate(schedule):
            print(f"\n📍 Scenario {i+1}/{len(schedule)}: {config['cutoff_date'].strftime('%Y-%m')} "
                  f"(Horizon: {config['forecast_horizon']}m)")
            
            # Prepare data for this scenario
            cutoff_idx = data_series.index.get_loc(config['cutoff_date'])
            train_data = data_series[:cutoff_idx + 1]
            test_data = data_series[cutoff_idx + 1:cutoff_idx + 1 + config['forecast_horizon']]
            
            # Filter competitive data for training period
            comp_data_filtered = None
            if competitive_data is not None:
                comp_data_filtered = competitive_data[
                    competitive_data['date'] <= config['cutoff_date']
                ].copy()
            
            # Test each model
            for model_name, model in models.items():
                result = self.train_and_evaluate_model(
                    model_name, model, train_data, test_data, comp_data_filtered, config
                )
                all_results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        self.results = results_df
        
        print(f"\n✅ Evaluation Complete! Generated {len(results_df)} results")
        return results_df
    
    def create_performance_matrix(self, metric='MAPE'):
        """
        Create performance matrix (cutoff date vs model)
        
        Args:
            metric: Performance metric to display
            
        Returns:
            Pivot table for visualization
        """
        if len(self.results) == 0:
            raise ValueError("No results available. Run evaluation first.")
        
        # Create pivot table
        matrix = self.results.pivot_table(
            values=metric,
            index=['cutoff_date', 'forecast_horizon'],
            columns='model',
            aggfunc='mean'
        )
        
        return matrix
    
    def generate_performance_report(self):
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary with summary statistics and insights
        """
        if len(self.results) == 0:
            raise ValueError("No results available. Run evaluation first.")
        
        report = {
            'summary_stats': {},
            'best_performers': {},
            'temporal_stability': {},
            'horizon_effects': {},
            'insights': []
        }
        
        # Summary statistics by model
        for model in self.results['model'].unique():
            model_data = self.results[self.results['model'] == model]
            
            report['summary_stats'][model] = {
                'mean_mape': model_data['MAPE'].mean(),
                'std_mape': model_data['MAPE'].std(),
                'mean_r2': model_data['R2'].mean(),
                'success_rate': (1 - model_data['MAPE'].isna().sum() / len(model_data)) * 100,
                'scenarios_tested': len(model_data)
            }
        
        # Best performers by metric
        metrics = ['MAPE', 'RMSE', 'R2', 'Directional_Accuracy']
        for metric in metrics:
            if metric in ['MAPE', 'RMSE']:
                best_model = self.results.groupby('model')[metric].mean().idxmin()
                best_value = self.results.groupby('model')[metric].mean().min()
            else:
                best_model = self.results.groupby('model')[metric].mean().idxmax()
                best_value = self.results.groupby('model')[metric].mean().max()
            
            report['best_performers'][metric] = {
                'model': best_model,
                'value': best_value
            }
        
        # Temporal stability (coefficient of variation)
        for model in self.results['model'].unique():
            model_data = self.results[self.results['model'] == model]
            cv_mape = model_data['MAPE'].std() / model_data['MAPE'].mean()
            
            report['temporal_stability'][model] = {
                'cv_mape': cv_mape,
                'stability_score': max(0, 1 - cv_mape)  # Higher is more stable
            }
        
        # Horizon effects
        horizon_performance = self.results.groupby(['model', 'forecast_horizon'])['MAPE'].mean().unstack()
        
        for model in horizon_performance.index:
            model_horizons = horizon_performance.loc[model]
            report['horizon_effects'][model] = {
                'best_horizon': model_horizons.idxmin(),
                'worst_horizon': model_horizons.idxmax(),
                'horizon_degradation': model_horizons.max() - model_horizons.min()
            }
        
        # Generate insights
        insights = []
        
        # Most stable model
        most_stable = min(report['temporal_stability'].items(), 
                         key=lambda x: x[1]['cv_mape'])[0]
        insights.append(f"Most temporally stable model: {most_stable}")
        
        # Best overall performer
        best_overall = report['best_performers']['MAPE']['model']
        insights.append(f"Best overall performer (MAPE): {best_overall}")
        
        # Horizon sensitivity
        horizon_sensitive_models = []
        for model, effects in report['horizon_effects'].items():
            if effects['horizon_degradation'] > 10:  # 10% MAPE degradation
                horizon_sensitive_models.append(model)
        
        if horizon_sensitive_models:
            insights.append(f"Horizon-sensitive models: {', '.join(horizon_sensitive_models)}")
        
        report['insights'] = insights
        
        return report
    
    def create_visualizations(self, save_plots=True):
        """
        Create comprehensive visualization suite
        
        Args:
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary of matplotlib figures
        """
        if len(self.results) == 0:
            raise ValueError("No results available. Run evaluation first.")
        
        figures = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance Heatmap (MAPE)
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        
        # Create heatmap data
        heatmap_data = self.results.pivot_table(
            values='MAPE',
            index='cutoff_date',
            columns=['model', 'forecast_horizon'],
            aggfunc='mean'
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   ax=ax1, cbar_kws={'label': 'MAPE (%)'})
        ax1.set_title('Model Performance Heatmap - MAPE (%)\nLower is Better', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Model & Forecast Horizon')
        ax1.set_ylabel('Cutoff Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        figures['heatmap_mape'] = fig1
        
        # 2. Temporal Performance Lines
        fig2, axes2 = plt.subplots(2, 2, figsize=(20, 12))
        
        metrics = ['MAPE', 'R2', 'RMSE', 'Directional_Accuracy']
        for i, metric in enumerate(metrics):
            ax = axes2[i // 2, i % 2]
            
            for model in self.results['model'].unique():
                model_data = self.results[self.results['model'] == model]
                temporal_perf = model_data.groupby('cutoff_date')[metric].mean()
                
                ax.plot(temporal_perf.index, temporal_perf.values, 
                       marker='o', linewidth=2, label=model)
            
            ax.set_title(f'{metric} Over Time')
            ax.set_xlabel('Cutoff Date')
            ax.set_ylabel(metric)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # MAPE - lower is better
                ax.set_ylim(bottom=0)
        
        plt.suptitle('Model Performance Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        figures['temporal_performance'] = fig2
        
        # 3. Horizon Effects
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        
        horizon_data = self.results.groupby(['model', 'forecast_horizon'])['MAPE'].mean().unstack()
        horizon_data.plot(kind='bar', ax=ax3, width=0.8)
        
        ax3.set_title('Model Performance by Forecast Horizon', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('MAPE (%)')
        ax3.legend(title='Forecast Horizon (months)')
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        figures['horizon_effects'] = fig3
        
        # 4. Growth Capture Analysis
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        
        growth_data = self.results.groupby('model')['growth_capture_pct'].mean().sort_values(ascending=False)
        bars = ax4.bar(range(len(growth_data)), growth_data.values)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if growth_data.iloc[i] > 90:
                bar.set_color('green')
            elif growth_data.iloc[i] > 70:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax4.set_title('Growth Trend Capture by Model\n(% of Actual Growth Captured)', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Growth Capture (%)')
        ax4.set_xticks(range(len(growth_data)))
        ax4.set_xticklabels(growth_data.index, rotation=45)
        ax4.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect Capture')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        
        figures['growth_capture'] = fig4
        
        # 5. Model Stability Analysis
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        
        stability_data = []
        for model in self.results['model'].unique():
            model_data = self.results[self.results['model'] == model]
            mean_mape = model_data['MAPE'].mean()
            std_mape = model_data['MAPE'].std()
            cv = std_mape / mean_mape if mean_mape > 0 else np.inf
            
            stability_data.append({
                'model': model,
                'mean_mape': mean_mape,
                'cv_mape': cv,
                'stability_score': max(0, 1 - cv)
            })
        
        stability_df = pd.DataFrame(stability_data)
        
        scatter = ax5.scatter(stability_df['mean_mape'], stability_df['cv_mape'], 
                            s=100, alpha=0.7, c=stability_df['stability_score'], 
                            cmap='RdYlGn')
        
        for i, model in enumerate(stability_df['model']):
            ax5.annotate(model, (stability_df.iloc[i]['mean_mape'], 
                               stability_df.iloc[i]['cv_mape']), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax5.set_title('Model Performance vs Stability\n(Lower-Left is Better)', 
                     fontsize=14, fontweight='bold')
        ax5.set_xlabel('Mean MAPE (%)')
        ax5.set_ylabel('Coefficient of Variation (MAPE)')
        plt.colorbar(scatter, label='Stability Score')
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        
        figures['stability_analysis'] = fig5
        
        # Save plots if requested
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for name, fig in figures.items():
                filename = f'evaluation_{name}_{timestamp}.png'
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"📊 Saved {filename}")
        
        return figures

def main():
    """Main execution function"""
    
    print("🚀 MEDIS Automated Model Evaluation")
    print("=" * 60)
    
    # Load and prepare data
    print("\n📂 Loading data...")
    try:
        df = load_and_prepare_data('MEDIS_VENTES.xlsx')
        medis_df = df[df['laboratoire'] == 'MEDIS'].copy()
        competitive_data = df[df['laboratoire'] != 'MEDIS'].copy()
        
        # Create monthly time series
        monthly_sales = medis_df.groupby('date')['sales'].sum().fillna(0).sort_index()
        
        print(f"✅ Loaded data: {len(monthly_sales)} months ({monthly_sales.index[0]} to {monthly_sales.index[-1]})")
        print(f"✅ Competitive data: {len(competitive_data):,} records")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize validator
    validator = WalkForwardValidator(min_train_months=36, max_test_months=12)
    
    # Run comprehensive evaluation
    print(f"\n🔬 Starting comprehensive evaluation...")
    
    # Test subset of models for speed (you can add more)
    models_to_test = ['Enhanced_LSTM', 'Transformer', 'Prophet', 'XGBoost', 'Naive']
    
    results_df = validator.run_comprehensive_evaluation(
        data_series=monthly_sales,
        competitive_data=competitive_data,
        models_to_test=models_to_test
    )
    
    # Generate performance report
    print(f"\n📊 Generating performance report...")
    report = validator.generate_performance_report()
    
    # Display key findings
    print(f"\n🏆 KEY FINDINGS:")
    print("=" * 40)
    for insight in report['insights']:
        print(f"• {insight}")
    
    print(f"\n📈 BEST PERFORMERS:")
    for metric, data in report['best_performers'].items():
        print(f"• {metric}: {data['model']} ({data['value']:.2f})")
    
    print(f"\n⚖️ TEMPORAL STABILITY RANKING:")
    stability_ranking = sorted(report['temporal_stability'].items(), 
                             key=lambda x: x[1]['stability_score'], reverse=True)
    for i, (model, data) in enumerate(stability_ranking[:5]):
        print(f"{i+1}. {model}: {data['stability_score']:.3f}")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    figures = validator.create_visualizations(save_plots=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f'detailed_evaluation_results_{timestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"💾 Saved detailed results: {results_filename}")
    
    # Save summary report
    report_filename = f'evaluation_report_{timestamp}.json'
    import json
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj
    
    # Clean report for JSON
    clean_report = {}
    for key, value in report.items():
        if isinstance(value, dict):
            clean_report[key] = {k: convert_numpy(v) for k, v in value.items()}
        else:
            clean_report[key] = convert_numpy(value)
    
    with open(report_filename, 'w') as f:
        json.dump(clean_report, f, indent=2, default=str)
    print(f"💾 Saved report: {report_filename}")
    
    # Display plots
    plt.show()
    
    print(f"\n✅ Evaluation complete! Check the generated files and visualizations.")

if __name__ == "__main__":
    main() 