# MEDIS Pharmaceutical Sales Forecasting Project

## 🎯 Project Overview

This project aims to create the best forecasting model for **MEDIS laboratory sales**, specifically for ATOR products (cholesterol medication), taking into account competitive dynamics and market trends.

### Key Features:
- 📊 **Comprehensive Data Analysis**: 7+ years of pharmaceutical sales data
- 🏆 **Competitive Intelligence**: Multi-competitor environment analysis  
- 🔮 **Advanced Forecasting**: Hierarchical time series models with competitive features
- 📈 **Interactive Dashboard**: Streamlit-based visualization and forecasting interface

## 📁 Project Structure

```
ventes-medis/
├── MEDIS_VENTES.xlsx              # Main dataset (pharmaceutical sales data)
├── medis_sales_analysis.ipynb     # Comprehensive data exploration notebook
├── streamlit_dashboard.py         # Interactive dashboard with ML forecasting tab
├── forecasting_models.py          # Core ML models and feature engineering
├── project_setup.md               # Detailed project strategy and approach
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── venv/                          # Virtual environment
```

## 🚀 Quick Start

### 1. Environment Setup

The project uses a Python virtual environment with essential data science libraries:

```bash
# Virtual environment is already created and configured
source venv/bin/activate

# Dependencies are already installed from requirements.txt
```

### 2. Data Exploration

Launch Jupyter to explore the pharmaceutical sales data:

```bash
# Activate environment and start jupyter
source venv/bin/activate
jupyter notebook medis_sales_analysis.ipynb
```

### 3. Interactive Dashboard

Run the Streamlit dashboard for interactive analysis:

```bash
# Launch the dashboard
source venv/bin/activate
streamlit run streamlit_dashboard.py
```

## 📊 Dataset Description

**File**: `MEDIS_VENTES.xlsx`
- **Records**: 7,424 transactions
- **Time Period**: April 2018 - April 2025 (monthly data)
- **Competitors**: 15+ pharmaceutical laboratories
- **Product**: ATOR (cholesterol medication)
- **Market Segments**: 4 dosage categories (10mg, 20mg, 40mg, 80mg)

### Key Columns:
- `laboratoire`: Pharmaceutical manufacturer (MEDIS + competitors)
- `PRODUIT`: Commercial product name
- `SOUS_MARCHE`: Sub-market grouping by dosage
- `PACK`: Package size (tablets per box)
- `ANNEE_MOIS`: Reference month (YYYYMM)
- `VENTE_IMS`: Monthly sales volume (IMS pharmacy estimates)

## 🔬 Analysis Capabilities

### Current Features:
✅ **Data Loading & Preprocessing**: Automated data cleaning and standardization  
✅ **Competitive Landscape**: Market share analysis across competitors  
✅ **Time Series Visualization**: Sales trends and seasonal patterns  
✅ **Interactive Filtering**: Dynamic dashboard with date/competitor filters  
✅ **Export Functions**: CSV data download capabilities  
✅ **ML Foundation**: Core forecasting pipeline with feature engineering
✅ **Baseline Models**: Naive, seasonal, and moving average forecasting
✅ **Model Evaluation**: Comprehensive metrics (MAPE, RMSE, directional accuracy)
✅ **Two-Tab Dashboard**: Data analysis + ML forecasting interface

### Planned Features:
🚧 **Prophet Forecasting**: Time series forecasting with competitive regressors  
🚧 **XGBoost Models**: Machine learning with engineered features  
🚧 **Hierarchical Forecasting**: Multi-level predictions (total → sub-market → package)  
🚧 **Model Ensemble**: Combined forecasting approaches  
🚧 **Performance Monitoring**: Real-time model accuracy tracking  

## 📈 Forecasting Strategy

Based on the comprehensive analysis in `project_setup.md`, our approach includes:

### 1. Hierarchical Time Series Forecasting
- **Top Level**: Total MEDIS sales across all products
- **Middle Level**: Sales by dosage category
- **Bottom Level**: Individual product-package combinations

### 2. Competitive Response Modeling
- Market share dynamics using attraction models
- Cross-competitor effects via vector autoregression
- Competitive action impact analysis

### 3. Advanced Model Ensemble
- **Prophet**: Seasonal patterns with competitive regressors
- **XGBoost**: Non-linear relationships and feature interactions
- **Statistical Models**: Economic relationships and cointegration

## 🛠️ Technical Stack

### Core Libraries:
- **Data**: `pandas`, `numpy`, `openpyxl`
- **Visualization**: `matplotlib`, `seaborn`
- **Statistics**: `statsmodels`, `scipy`
- **Dashboard**: `streamlit`
- **Notebooks**: `jupyter`, `ipython`

### Future Additions:
- `prophet` - Facebook's time series forecasting
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting models
- `plotly` - Interactive visualizations

## 📱 Dashboard Features

The Streamlit dashboard provides:

### 📊 Key Metrics Dashboard
- Total sales volumes
- MEDIS market share
- Active competitor count
- Performance indicators

### 📈 Interactive Visualizations
- Time series plots with filtering
- Competitive landscape analysis
- Sub-market distribution charts
- Market share evolution

### 🔍 Dynamic Filtering
- Date range selection
- Laboratory-specific views
- Sub-market focus
- Package size analysis

## 🚀 Next Development Phases

### Phase 1: Baseline Models (2-3 weeks)
- [ ] ARIMA and exponential smoothing models
- [ ] Seasonal decomposition analysis
- [ ] Basic competitive correlation features

### Phase 2: Advanced Models (4-6 weeks)
- [ ] Prophet with competitive regressors
- [ ] XGBoost with engineered features
- [ ] LSTM neural networks for long-term dependencies

### Phase 3: Production Deployment (2-3 weeks)
- [ ] Model ensemble framework
- [ ] Performance monitoring system
- [ ] Production-ready forecasting pipeline

## 📋 Usage Examples

### Basic Data Exploration
```python
# Load and explore data
df = pd.read_excel('MEDIS_VENTES.xlsx', sheet_name='Data')
medis_data = df[df['laboratoire'] == 'MEDIS']

# Quick market share analysis
market_share = medis_data['VENTE_IMS'].sum() / df['VENTE_IMS'].sum()
print(f"MEDIS Market Share: {market_share:.1%}")
```

### Dashboard Launch
```bash
# Start the interactive dashboard (includes ML forecasting tab)
streamlit run streamlit_dashboard.py
```

### ML Models Testing
```python
# Test the forecasting pipeline
from forecasting_models import load_and_prepare_data, FeatureEngineer, BaselineModels

# Load data
data = load_and_prepare_data()

# Create features
fe = FeatureEngineer(data)
enhanced_data = fe.create_temporal_features(data)

# Test baseline models
baseline = BaselineModels()
forecast = baseline.seasonal_naive(medis_sales_series, periods=12)
```

## 🤝 Contributing

This project follows a structured development approach:

1. **Data Analysis**: Comprehensive exploration in Jupyter notebooks
2. **Model Development**: Systematic forecasting algorithm implementation
3. **Dashboard Creation**: Interactive Streamlit applications
4. **Documentation**: Clear documentation and usage examples

## 📞 Contact & Support

For questions about the MEDIS pharmaceutical sales forecasting project:
- Review the detailed strategy in `project_setup.md`
- Explore the data analysis in `medis_sales_analysis.ipynb`
- Test the dashboard with `streamlit run streamlit_dashboard.py`

---

*Built with ❤️ for pharmaceutical sales forecasting and competitive intelligence* 