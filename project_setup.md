File Structure Overview
The Excel file contains several sheets with pharmaceutical sales data:
Sheet: Feuil2
Contains summary data for ATOR products with monthly sales figures
Covers the period from April 2018 to April 2025
Shows two main product categories: 30mg and 90mg formulations
Monthly sales volumes ranging from around 2,000 to over 11,000 units
Sheet: Feuil6
Provides a comprehensive product summary across different package sizes (30, 60, 90)
Lists various pharmaceutical products including:
ATOR, ATROVAST, CHOLDOWN, CHOLESTYL
CRESTOR, LAPTOR, NEACREST, ROCOR
ROSACT, ROSULIP, ROSUVANS, ROSUVASCOR
STATINOR, SUPERSTAT, TAHOR, TAVASTOR
TORVA, VISSEN, and others
Shows total sales figures for each product
Sheet: Data (Main Dataset)
22,000+ rows of detailed transactional data
16 columns including:
Laboratory information (MEDIS, PHARMACARE, SOCIETE THERA SA., etc.)
Product details (names, presentations, dosages)
Market segments and package sizes
Monthly sales data from 2018-2025
Stock quantities and rupture days
Sales figures and quantities
Key Data Points
Time Period: April 2018 to April 2025
Main Companies: MEDIS, PHARMACARE, SOCIETE THERA SA., GALPHARMA, TAHA PHARMA, ACT PHARMA, ADVANS PHARMA SA., TERIAK
Product Categories: Various cholesterol-lowering medications (statins) in different dosages and package sizes
Geographic Market: Appears to be pharmaceutical sales data, likely from a specific region or country
Data Quality
The dataset appears well-structured with:
Consistent date formatting (YYYYMM)
Standardized product codes and names
Complete sales figures for most entries
Some missing values in stock and rupture data for earlier periods
Recent data (2024-2025) includes more detailed inventory metrics
This Excel file contains rich pharmaceutical market data that would be suitable for various analyses including sales trends, market share analysis, inventory management, and competitive intelligence.
Great. Listen up: Tu vas trouver ci-joint l'historique des ventes de produits MEDIS ainsi que des concurrents sur quelques années pour le produit ATOR qui est prescrit pour l'hypercholestérolémie. La feuille Data contient les données, ci-dessous la description de chaque colonne : laboratoire : Le fabricant du médicament PRODUIT : Le nom commercial du produit PRESENTATION : Le nom complet du produit. exemple ATOR CPR. PELLIC. 10 MG 28 : c'est le produit ATOR avec la forme comprimé, dosage 10 et boite de 28 comprimés. SOUS_MARCHE : regroupe les produits avec le même dosage. C'est-à dire les concurrents sur le même segment. PACK : C'est le nombre de comprimés par boites. Dans le marché ATOR c'est essentiellement des boîtes de 30 ou de 90. Tu peux arrondir tous les packages à 30/60 ou 90. ANNEE_MOIS : Mois de référence. VENTE_IMS : Nombre de boites vendues par mois selon la société IMS qui estime ce chiffre en analysant les ventes réalisées au niveau des officines (pharmacies) VENTE_USINE : c'est le nombre de boites vendues par MEDIS directement aux grossistes (utilisées à titre indicatif). L'objectif est de créer le meilleur modèle de prévision pour les ventes du laboratoire MEDIS en tenant compte de l'historique des dernières années et de l'historique des concurrents par sous-marché.   I want you to make a very deep research and help me identify the best ways I should try to tackle this problem.
Great. Listen up: Tu vas trouver ci-joint l'historique des ventes de produits MEDIS ainsi que des concurrents sur quelques années pour le produit ATOR qui est prescrit pour l'hypercholestérolémie. La feuille Data contient les données, ci-dessous la description de chaque colonne : laboratoire : Le fabricant du médicament PRODUIT : Le nom commercial du produit PRESENTATION : Le nom complet du produit. exemple ATOR CPR. PELLIC. 10 MG 28 : c'est le produit ATOR avec la forme comprimé, dosage 10 et boite de 28 comprimés. SOUS_MARCHE : regroupe les produits avec le même dosage. C'est-à dire les concurrents sur le même segment. PACK : C'est le nombre de comprimés par boites. Dans le marché ATOR c'est essentiellement des boîtes de 30 ou de 90. Tu peux arrondir tous les packages à 30/60 ou 90. ANNEE_MOIS : Mois de référence. VENTE_IMS : Nombre de boites vendues par mois selon la société IMS qui estime ce chiffre en analysant les ventes réalisées au niveau des officines (pharmacies) VENTE_USINE : c'est le nombre de boites vendues par MEDIS directement aux grossistes (utilisées à titre indicatif). L'objectif est de créer le meilleur modèle de prévision pour les ventes du laboratoire MEDIS en tenant compte de l'historique des dernières années et de l'historique des concurrents par sous-marché. I want you to make a very deep research and help me identify the best ways I should try to tackle this problem.
Strategic Approaches for MEDIS Sales Forecasting: A Comprehensive Analysis

Based on your pharmaceutical sales data for ATOR products, I'll provide a deep analysis of the best forecasting approaches to predict MEDIS sales while accounting for competitive dynamics and market trends.
Understanding the Forecasting Challenge
Data Characteristics Analysis
Your dataset presents several unique characteristics that will influence the modeling approach:
Multi-competitor environment: MEDIS competes with 8+ laboratories (PHARMACARE, SOCIETE THERA SA, GALPHARMA, etc.)
Product segmentation: Four main dosage categories (10mg, 20mg, 40mg, 80mg) with different competitive dynamics
Package variations: Primarily 30 and 90-tablet packages, requiring aggregation or separate modeling
Time span: 7 years of monthly data (2018-2025) providing sufficient historical depth
Market maturity: ATOR is an established cholesterol medication in a mature therapeutic area
Key Forecasting Considerations
Market Dynamics:
Seasonal patterns in pharmaceutical consumption
Competitor entry/exit effects
Market share evolution over time
Price competition impacts
Regulatory Environment:
Prescription patterns and medical guidelines
Generic competition effects
Reimbursement policy changes
Recommended Forecasting Approaches
1. Hierarchical Time Series Forecasting
Why This Approach:
Given your multi-level data structure (laboratory → product → dosage → package), hierarchical forecasting is ideal.
Implementation Strategy:
Top Level: Total MEDIS sales across all ATOR products
Middle Level: Sales by dosage category (10mg, 20mg, 40mg, 80mg)
Bottom Level: Individual product-package combinations
Techniques to Consider:
Top-down reconciliation: Forecast total sales, then disaggregate
Bottom-up aggregation: Forecast individual SKUs, then sum up
Middle-out approach: Forecast at dosage level, then reconcile up and down
2. Competitive Response Modeling
Market Share Dynamics:
Model MEDIS share within each sub-market using attraction models
Account for competitive actions and market responses
Implement vector autoregression (VAR) to capture cross-competitor effects
Recommended Models:
Multinomial Logit Market Share Model: Captures competitive dynamics
Bass Diffusion Model: For new product launches or market expansion
Dirichlet-Multinomial Model: For brand switching behavior
3. Advanced Time Series Ensemble Methods
Primary Recommendations:
Prophet with Competitive Regressors:
Handles seasonality and trend changes well
Add competitor sales as external regressors
Captures market growth/decline patterns
Robust to missing data and outliers
LSTM Neural Networks:
Excellent for capturing long-term dependencies
Can incorporate multiple input features simultaneously
Handles non-linear relationships between variables
Suitable for multivariate time series
XGBoost with Time Series Features:
Strong performance on tabular data
Easy to interpret feature importance
Handles categorical variables (laboratory, product) well
Fast training and prediction
4. Hybrid Econometric-ML Approach
Econometric Foundation:
Vector Error Correction Model (VECM) for long-term equilibrium relationships
Capture cointegration between MEDIS and competitor sales
Model price elasticity and cross-price effects
Machine Learning Enhancement:
Use ML models to capture non-linear residual patterns
Ensemble econometric predictions with ML forecasts
Dynamic model selection based on market conditions
Feature Engineering Strategy
Core Variables
Temporal Features:
Month/quarter indicators for seasonality
Year-over-year growth rates
Moving averages (3, 6, 12 months)
Trend and seasonal decomposition components
Competitive Intelligence:
Competitor market share by sub-market
Competitive intensity index (number of active competitors)
Price positioning relative to competitors
New product launch indicators
Market Context:
Total market size by dosage category
Market concentration ratios (HHI index)
Competitive response lags
Market maturity indicators
Advanced Feature Engineering
Interaction Effects:
MEDIS performance vs. specific competitor actions
Dosage-specific competitive responses
Package size preference shifts
Seasonal-competitive interactions
Leading Indicators:
Competitor inventory levels (from QTE_STOCK data)
Stock-out frequency (NBJ_RUPTURE)
Market entry/exit patterns
Regulatory approval timelines
Model Selection Framework
Phase 1: Baseline Models
Naive forecasting (last year same month)
Exponential smoothing (Holt-Winters)
ARIMA models for each product line
Linear regression with seasonal dummies
Phase 2: Advanced Single Models
Prophet with competitive regressors
LSTM neural networks
XGBoost with engineered features
Structural time series models
Phase 3: Ensemble Methods
Weighted averaging of top-performing models
Stacking with meta-learner
Dynamic model selection based on market conditions
Bayesian model averaging
Validation and Evaluation Strategy
Cross-Validation Design
Time series split: Respect temporal order
Walk-forward validation: Simulate real-world deployment
Competitor-aware splits: Ensure competitive dynamics are preserved
Performance Metrics
Accuracy: MAPE, RMSE, MAE
Directional accuracy: Trend prediction success
Business metrics: Revenue impact, inventory optimization
Robustness: Performance across different market conditions
Backtesting Framework
Out-of-sample testing: 12-18 months holdout
Stress testing: Performance during market disruptions
Scenario analysis: Different competitive landscapes
Sensitivity analysis: Impact of key assumptions
Implementation Roadmap
Phase 1: Data Preparation (2-3 weeks)
Clean and validate historical data
Create hierarchical data structure
Engineer baseline features
Establish data pipeline
Phase 2: Model Development (4-6 weeks)
Implement baseline models
Develop advanced single models
Create ensemble framework
Validate performance
Phase 3: Deployment (2-3 weeks)
Production pipeline setup
Monitoring and alerting
Regular model retraining
Stakeholder training
Risk Mitigation Strategies
Model Risk
Ensemble diversity: Combine different model types
Regular retraining: Adapt to changing market conditions
Human oversight: Expert review of forecasts
Scenario planning: Multiple forecast scenarios
Data Quality
Anomaly detection: Identify unusual patterns
Data validation: Consistency checks
Missing data handling: Robust imputation methods
Competitor data verification: External validation
Market Risk
Competitive monitoring: Track competitor activities
Regulatory tracking: Monitor policy changes
Economic indicators: Include macroeconomic factors
Expert judgment: Incorporate market intelligence
Success Metrics and KPIs
Forecast Accuracy
Monthly MAPE: < 15% for total MEDIS sales
Quarterly MAPE: < 10% for aggregated forecasts
Directional accuracy: > 80% for trend predictions
Business Impact
Inventory optimization: Reduce stock-outs by 25%
Revenue planning: Improve budget accuracy by 20%
Competitive response: Faster reaction to market changes
Strategic planning: Better long-term market position
Conclusion
The optimal approach for MEDIS sales forecasting should combine hierarchical time series methods with competitive intelligence, implemented through an ensemble of econometric and machine learning models. The key to success lies in capturing competitive dynamics while maintaining forecast accuracy across different time horizons and market conditions.
Start with Prophet and XGBoost as your core models, then gradually incorporate more sophisticated approaches like LSTM networks and competitive response models. The hierarchical structure will ensure consistency across product lines while the ensemble approach will provide robustness against model uncertainty.