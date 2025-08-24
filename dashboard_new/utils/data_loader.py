"""
MEDIS Pharmaceutical Sales Data Loader

This module handles all data loading and preprocessing for the MEDIS sales forecasting dashboard.
It provides efficient, cached data loading with proper error handling and data validation.

Author: AI Assistant
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import streamlit as st

# Global cache for data loading
@st.cache_data
def _load_excel_data(file_path: str) -> pd.DataFrame:
    """Cached function to load Excel data"""
    # Convert relative path to absolute path if needed
    if not file_path.startswith('/'):
        import os
        file_path = os.path.abspath(file_path)
    return pd.read_excel(file_path, sheet_name='Data')

class MedisDataLoader:
    """
    Handles loading and preprocessing of MEDIS pharmaceutical sales data.

    This class provides:
    - Efficient data loading with caching
    - Data validation and quality checks
    - Standardized preprocessing pipeline
    - Multiple data views (raw, aggregated, time series)
    """

    def __init__(self, file_path: str = '../MEDIS_VENTES.xlsx'):
        """
        Initialize the data loader.

        Args:
            file_path: Path to the Excel file containing sales data
        """
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.metadata = {}

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from Excel file with caching.

        Returns:
            Raw pandas DataFrame from the Data sheet

        Raises:
            FileNotFoundError: If the Excel file is not found
            ValueError: If the Data sheet is not found
        """
        try:
            print(f"🔄 Loading data from {self.file_path}...")

            # Use cached data loading function
            df = _load_excel_data(self.file_path)

            # Basic validation
            required_columns = ['laboratoire', 'PRODUIT', 'SOUS_MARCHE', 'PACK', 'ANNEE_MOIS', 'VENTE_IMS']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
            self.raw_data = df
            return df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw data for analysis and forecasting.

        Args:
            df: Raw DataFrame from load_raw_data()

        Returns:
            Preprocessed DataFrame with additional features
        """
        print("🔄 Preprocessing data...")

        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Convert date column
        processed_df['date'] = pd.to_datetime(processed_df['ANNEE_MOIS'].astype(str), format='%Y%m')

        # Extract temporal features
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['quarter'] = processed_df['date'].dt.quarter

        # Standardize sales column
        processed_df['sales'] = processed_df['VENTE_IMS'].fillna(0)

        # Standardize package sizes (round to 30/60/90)
        def standardize_package_size(pack):
            if pd.isna(pack):
                return pack
            if pack <= 35:
                return 30
            elif pack <= 70:
                return 60
            else:
                return 90

        processed_df['pack_std'] = processed_df['PACK'].apply(standardize_package_size)

        # Add derived features
        processed_df['is_medis'] = (processed_df['laboratoire'] == 'MEDIS').astype(int)

        print("✅ Data preprocessing complete")
        self.processed_data = processed_df
        return processed_df

    def get_data(self) -> pd.DataFrame:
        """
        Get the fully processed data.

        Returns:
            Fully processed DataFrame ready for analysis
        """
        if self.processed_data is None:
            raw_data = self.load_raw_data()
            self.processed_data = self.preprocess_data(raw_data)

        return self.processed_data

    def get_medis_data(self) -> pd.DataFrame:
        """
        Get only MEDIS data.

        Returns:
            DataFrame filtered to only MEDIS laboratory
        """
        data = self.get_data()
        return data[data['laboratoire'] == 'MEDIS'].copy()

    def get_competitor_data(self) -> pd.DataFrame:
        """
        Get only competitor data (excluding MEDIS).

        Returns:
            DataFrame with all competitors except MEDIS
        """
        data = self.get_data()
        return data[data['laboratoire'] != 'MEDIS'].copy()

    def get_monthly_sales(self, laboratory: Optional[str] = None) -> pd.DataFrame:
        """
        Get monthly sales aggregated data.

        Args:
            laboratory: Specific laboratory to filter (None for all)

        Returns:
            Monthly aggregated sales data
        """
        data = self.get_data()

        if laboratory:
            data = data[data['laboratoire'] == laboratory]

        monthly = data.groupby('date')['sales'].sum().reset_index()
        monthly = monthly.sort_values('date')

        return monthly

    def get_market_share_data(self) -> pd.DataFrame:
        """
        Calculate market share by laboratory over time.

        Returns:
            DataFrame with market share percentages by date and laboratory
        """
        data = self.get_data()

        # Calculate total market sales by month
        total_sales = data.groupby('date')['sales'].sum().reset_index()
        total_sales = total_sales.rename(columns={'sales': 'total_market_sales'})

        # Calculate market share
        market_share = data.groupby(['date', 'laboratoire'])['sales'].sum().reset_index()
        market_share = market_share.merge(total_sales, on='date')
        market_share['market_share'] = market_share['sales'] / market_share['total_market_sales'] * 100

        return market_share

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the dataset.

        Returns:
            Dictionary with dataset metadata
        """
        if not self.metadata:
            data = self.get_data()

            self.metadata = {
                'total_records': len(data),
                'date_range': {
                    'start': data['date'].min(),
                    'end': data['date'].max()
                },
                'laboratories': data['laboratoire'].nunique(),
                'products': data['PRODUIT'].nunique(),
                'market_segments': data['SOUS_MARCHE'].nunique(),
                'total_sales': data['sales'].sum(),
                'medis_market_share': (data[data['laboratoire'] == 'MEDIS']['sales'].sum() /
                                     data['sales'].sum() * 100)
            }

        return self.metadata

    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks.

        Returns:
            Dictionary with data quality metrics
        """
        data = self.get_data()

        quality_report = {
            'total_records': len(data),
            'null_counts': data.isnull().sum().to_dict(),
            'duplicate_records': data.duplicated().sum(),
            'negative_sales': (data['sales'] < 0).sum(),
            'zero_sales': (data['sales'] == 0).sum(),
            'outliers': self._detect_outliers(data)
        }

        return quality_report

    def _detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> int:
        """
        Detect outliers in sales data using z-score.

        Args:
            df: DataFrame with sales data
            threshold: Z-score threshold for outlier detection

        Returns:
            Number of outliers detected
        """
        z_scores = np.abs((df['sales'] - df['sales'].mean()) / df['sales'].std())
        return (z_scores > threshold).sum()
