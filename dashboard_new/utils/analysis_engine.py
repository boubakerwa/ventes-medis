"""
MEDIS Pharmaceutical Sales Analysis Engine

This module provides comprehensive data analysis functions for the MEDIS sales forecasting dashboard.
It includes statistical analysis, competitive intelligence, and business insights.

Author: AI Assistant
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MedisAnalysisEngine:
    """
    Analysis engine for MEDIS pharmaceutical sales data.

    Provides comprehensive analysis including:
    - Competitive intelligence
    - Market share analysis
    - Growth trends
    - Seasonal patterns
    - Business metrics
    """

    def __init__(self, data_loader):
        """
        Initialize the analysis engine.

        Args:
            data_loader: Instance of MedisDataLoader
        """
        self.data_loader = data_loader

    def get_key_metrics(self) -> Dict[str, Any]:
        """
        Calculate key business metrics for dashboard display.

        Returns:
            Dictionary with key metrics
        """
        data = self.data_loader.get_data()

        # Basic metrics
        total_sales = data['sales'].sum()
        medis_sales = data[data['laboratoire'] == 'MEDIS']['sales'].sum()
        medis_market_share = (medis_sales / total_sales) * 100

        # Growth metrics
        monthly_sales = self.data_loader.get_monthly_sales()
        latest_sales = monthly_sales['sales'].iloc[-1]
        first_sales = monthly_sales['sales'].iloc[0]
        total_growth = ((latest_sales - first_sales) / first_sales) * 100

        # Competitive metrics
        active_competitors = data['laboratoire'].nunique()
        medis_competitors = active_competitors - 1  # Excluding MEDIS

        # Time period
        date_range = data['date'].min(), data['date'].max()
        months_active = ((date_range[1] - date_range[0]).days) / 30

        return {
            'total_sales': total_sales,
            'medis_sales': medis_sales,
            'medis_market_share': medis_market_share,
            'total_growth': total_growth,
            'active_competitors': medis_competitors,
            'months_active': int(months_active),
            'avg_monthly_sales': total_sales / months_active if months_active > 0 else 0,
            'peak_monthly_sales': monthly_sales['sales'].max(),
            'date_range': date_range
        }

    def get_competitive_analysis(self) -> Dict[str, Any]:
        """
        Analyze competitive landscape.

        Returns:
            Dictionary with competitive analysis results
        """
        data = self.data_loader.get_data()

        # Market share by laboratory
        market_share = data.groupby('laboratoire')['sales'].sum().sort_values(ascending=False)
        total_sales = market_share.sum()
        market_share_pct = (market_share / total_sales * 100).round(1)

        # MEDIS position
        medis_sales = market_share['MEDIS']
        medis_rank = market_share.index.get_loc('MEDIS') + 1

        # Top competitors
        top_competitors = market_share_pct.head(5).to_dict()

        # Market concentration (HHI - Herfindahl-Hirschman Index)
        hhi = (market_share_pct ** 2).sum()

        # Competitive dynamics over time
        market_share_ts = self.data_loader.get_market_share_data()

        # MEDIS trend
        medis_trend = market_share_ts[market_share_ts['laboratoire'] == 'MEDIS'].copy()
        medis_trend = medis_trend.sort_values('date')
        medis_trend['growth'] = medis_trend['market_share'].pct_change() * 100

        return {
            'market_share_by_lab': market_share_pct.to_dict(),
            'medis_rank': medis_rank,
            'top_competitors': top_competitors,
            'market_concentration_hhi': hhi,
            'medis_trend': medis_trend,
            'total_competitors': len(market_share) - 1
        }

    def get_growth_analysis(self) -> Dict[str, Any]:
        """
        Analyze growth patterns and trends.

        Returns:
            Dictionary with growth analysis results
        """
        data = self.data_loader.get_data()

        # Overall growth
        monthly_sales = self.data_loader.get_monthly_sales()
        monthly_sales = monthly_sales.set_index('date')

        # Calculate growth rates
        growth_analysis = {
            'monthly_growth': monthly_sales['sales'].pct_change().mean() * 100,
            'yearly_growth': monthly_sales['sales'].pct_change(12).mean() * 100,
            'volatility': monthly_sales['sales'].std() / monthly_sales['sales'].mean() * 100,
            'peak_month': monthly_sales['sales'].idxmax().strftime('%B %Y'),
            'lowest_month': monthly_sales['sales'].idxmin().strftime('%B %Y')
        }

        # MEDIS specific growth
        medis_data = self.data_loader.get_medis_data()
        medis_monthly = medis_data.groupby('date')['sales'].sum()

        medis_growth = {
            'starting_sales': medis_monthly.iloc[0],
            'current_sales': medis_monthly.iloc[-1],
            'total_growth': ((medis_monthly.iloc[-1] - medis_monthly.iloc[0]) / medis_monthly.iloc[0]) * 100,
            'avg_growth_per_year': medis_monthly.pct_change(12).mean() * 100,
            'peak_sales': medis_monthly.max(),
            'peak_date': medis_monthly.idxmax().strftime('%B %Y')
        }

        # Growth by segment
        segment_growth = data.groupby(['SOUS_MARCHE', 'laboratoire'])['sales'].sum().unstack()
        medis_segment_growth = segment_growth.loc[:, 'MEDIS'].sort_values(ascending=False)

        return {
            'overall_growth': growth_analysis,
            'medis_growth': medis_growth,
            'medis_segment_growth': medis_segment_growth.to_dict(),
            'growth_comparison': {
                'medis_vs_market': medis_growth['total_growth'] - growth_analysis['yearly_growth']
            }
        }

    def get_seasonal_analysis(self) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in sales.

        Returns:
            Dictionary with seasonal analysis results
        """
        data = self.data_loader.get_data()

        # Monthly patterns for all market
        monthly_patterns = data.groupby(data['date'].dt.month)['sales'].mean()
        monthly_patterns.index = pd.to_datetime(monthly_patterns.index, format='%m').strftime('%B')

        # MEDIS seasonal patterns
        medis_data = self.data_loader.get_medis_data()
        medis_monthly = medis_data.groupby(medis_data['date'].dt.month)['sales'].mean()
        medis_monthly.index = pd.to_datetime(medis_monthly.index, format='%m').strftime('%B')

        # Seasonal strength
        overall_seasonal_strength = monthly_patterns.std() / monthly_patterns.mean() * 100
        medis_seasonal_strength = medis_monthly.std() / medis_monthly.mean() * 100

        # Peak and low seasons
        peak_month = monthly_patterns.idxmax()
        low_month = monthly_patterns.idxmin()

        medis_peak_month = medis_monthly.idxmax()
        medis_low_month = medis_monthly.idxmin()

        return {
            'market_seasonal_pattern': monthly_patterns.to_dict(),
            'medis_seasonal_pattern': medis_monthly.to_dict(),
            'market_seasonal_strength': overall_seasonal_strength,
            'medis_seasonal_strength': medis_seasonal_strength,
            'peak_season': {
                'market': peak_month,
                'medis': medis_peak_month
            },
            'low_season': {
                'market': low_month,
                'medis': medis_low_month
            }
        }

    def get_product_analysis(self) -> Dict[str, Any]:
        """
        Analyze product performance and segments.

        Returns:
            Dictionary with product analysis results
        """
        data = self.data_loader.get_data()

        # Product performance
        product_sales = data.groupby('PRODUIT')['sales'].sum().sort_values(ascending=False)
        total_product_sales = product_sales.sum()
        product_share = (product_sales / total_product_sales * 100).round(1)

        # MEDIS products specifically
        medis_products = data[data['laboratoire'] == 'MEDIS']['PRODUIT'].unique().tolist()

        # Segment analysis (dosage categories)
        segment_sales = data.groupby('SOUS_MARCHE')['sales'].sum().sort_values(ascending=False)
        total_segment_sales = segment_sales.sum()
        segment_share = (segment_sales / total_segment_sales * 100).round(1)

        # Package size analysis
        package_sales = data.groupby('pack_std')['sales'].sum().sort_values(ascending=False)
        total_package_sales = package_sales.sum()
        package_share = (package_sales / total_package_sales * 100).round(1)

        # MEDIS vs competitors by segment
        segment_comparison = data.pivot_table(
            values='sales',
            index='SOUS_MARCHE',
            columns='laboratoire',
            aggfunc='sum',
            fill_value=0
        )

        # MEDIS segment share
        medis_segment_share = {}
        for segment in segment_comparison.index:
            segment_total = segment_comparison.loc[segment].sum()
            if segment_total > 0:
                medis_share = (segment_comparison.loc[segment, 'MEDIS'] / segment_total) * 100
                medis_segment_share[segment] = round(medis_share, 1)

        return {
            'product_performance': product_share.to_dict(),
            'segment_performance': segment_share.to_dict(),
            'package_performance': package_share.to_dict(),
            'medis_products': medis_products,
            'medis_segment_share': medis_segment_share,
            'top_products': product_sales.head(5).to_dict(),
            'top_segments': segment_sales.head(4).to_dict()
        }

    def get_business_insights(self) -> Dict[str, Any]:
        """
        Generate business insights and recommendations.

        Returns:
            Dictionary with business insights
        """
        metrics = self.get_key_metrics()
        competitive = self.get_competitive_analysis()
        growth = self.get_growth_analysis()
        seasonal = self.get_seasonal_analysis()
        products = self.get_product_analysis()

        insights = []

        # Market position insights
        if metrics['medis_market_share'] > 20:
            insights.append("🏆 MEDIS is a market leader with strong competitive position")
        elif metrics['medis_market_share'] > 10:
            insights.append("📈 MEDIS has a solid market position with growth potential")
        else:
            insights.append("📊 MEDIS has room for market share growth")

        # Growth insights
        if growth['medis_growth']['total_growth'] > 200:
            insights.append("🚀 Exceptional growth achieved - strong business momentum")
        elif growth['medis_growth']['total_growth'] > 100:
            insights.append("📈 Strong growth performance above market average")
        else:
            insights.append("📊 Moderate growth - opportunities for acceleration")

        # Seasonal insights
        if seasonal['medis_seasonal_strength'] > seasonal['market_seasonal_strength']:
            insights.append("📅 MEDIS shows stronger seasonal patterns than market average")
        else:
            insights.append("📊 MEDIS follows general market seasonal trends")

        # Competitive insights
        hhi = competitive['market_concentration_hhi']
        if hhi > 2500:
            insights.append("🏢 Highly concentrated market - established players dominate")
        elif hhi > 1500:
            insights.append("🏭 Moderately concentrated market - room for competition")
        else:
            insights.append("🌊 Fragmented market - opportunity for consolidation")

        # Product insights
        top_segment = max(products['medis_segment_share'].items(), key=lambda x: x[1])
        insights.append(f"💊 Strongest performance in {top_segment[0]} segment ({top_segment[1]}% share)")

        return {
            'insights': insights,
            'recommendations': [
                "Focus on high-performing dosage segments",
                "Leverage seasonal patterns for inventory planning",
                "Monitor competitive pricing strategies",
                "Consider product portfolio expansion",
                "Invest in market share growth opportunities"
            ],
            'risk_factors': [
                "Market saturation in mature segments",
                "Competitor entry in high-growth areas",
                "Regulatory changes affecting pricing",
                "Seasonal demand fluctuations",
                "Supply chain disruptions"
            ]
        }

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.

        Returns:
            Dictionary with complete analysis summary
        """
        return {
            'key_metrics': self.get_key_metrics(),
            'competitive_analysis': self.get_competitive_analysis(),
            'growth_analysis': self.get_growth_analysis(),
            'seasonal_analysis': self.get_seasonal_analysis(),
            'product_analysis': self.get_product_analysis(),
            'business_insights': self.get_business_insights()
        }
