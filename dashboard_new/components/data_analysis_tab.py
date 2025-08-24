"""
MEDIS Pharmaceutical Sales Data Analysis Tab

This module implements the comprehensive Data Analysis tab for the MEDIS sales forecasting dashboard.
It provides business-focused analytics with interactive visualizations and insights.

Author: AI Assistant
Date: 2025-01-09
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import MedisDataLoader
from utils.analysis_engine import MedisAnalysisEngine
from utils.visualization_utils import MedisVisualizationUtils

class DataAnalysisTab:
    """
    Data Analysis Tab for MEDIS Sales Forecasting Dashboard

    Features:
    - Executive overview with key metrics
    - Competitive intelligence
    - Growth and trend analysis
    - Seasonal pattern analysis
    - Product and market segment analysis
    - Business insights and recommendations
    """

    def __init__(self):
        """Initialize the Data Analysis Tab"""
        self.data_loader = None
        self.analysis_engine = None
        self.analysis_results = None

    def render(self):
        """
        Render the complete Data Analysis Tab
        """
        st.header("📊 Data Analysis & Business Intelligence")
        st.markdown("Comprehensive analysis of MEDIS pharmaceutical sales data")

        # Initialize components
        self._initialize_components()

        if self.analysis_results is None:
            self._load_and_analyze_data()

        # Render dashboard sections
        if self.analysis_results:
            self._render_executive_overview()
            self._render_competitive_analysis()
            self._render_growth_analysis()
            self._render_seasonal_analysis()
            self._render_product_analysis()
            self._render_data_quality_section()

    def _initialize_components(self):
        """Initialize data loader and analysis engine"""
        try:
            self.data_loader = MedisDataLoader()
            self.analysis_engine = MedisAnalysisEngine(self.data_loader)
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            st.stop()

    def _load_and_analyze_data(self):
        """Load and analyze data with progress tracking"""
        try:
            with st.spinner("🔄 Loading and analyzing data..."):
                # Load data
                self.data_loader.get_data()

                # Run comprehensive analysis
                self.analysis_results = self.analysis_engine.get_summary_report()

                st.success("✅ Data analysis complete!")

        except Exception as e:
            st.error(f"Error during data analysis: {e}")
            st.stop()

    def _render_executive_overview(self):
        """Render executive overview section"""
        st.subheader("📈 Executive Overview")

        # Key metrics cards
        metrics = self.analysis_results['key_metrics']
        MedisVisualizationUtils.create_metric_cards(metrics)

        st.markdown("---")

        # Data period information
        col1, col2, col3 = st.columns(3)

        with col1:
            date_range = metrics['date_range']
            st.info(f"📅 **Analysis Period**\n\n{date_range[0].strftime('%B %Y')} to {date_range[1].strftime('%B %Y')}\n\n({metrics['months_active']} months)")

        with col2:
            market_share = metrics['medis_market_share']
            rank_color = "🟢" if market_share > 25 else "🟡" if market_share > 15 else "🔴"
            st.info(f"🏆 **Market Position**\n\n{rank_color} **{market_share:.1f}%** Market Share\n\n#{self.analysis_results['competitive_analysis']['medis_rank']} in market")

        with col3:
            growth = metrics['total_growth']
            growth_color = "🟢" if growth > 200 else "🟡" if growth > 100 else "🔴"
            st.info(f"📈 **Growth Performance**\n\n{growth_color} **{growth:+.1f}%** Total Growth\n\n({metrics['avg_monthly_sales']:,.0f} avg monthly sales)")

    def _render_competitive_analysis(self):
        """Render competitive analysis section"""
        st.subheader("🏆 Competitive Intelligence")

        competitive = self.analysis_results['competitive_analysis']

        # Market share charts
        col1, col2 = st.columns([2, 3])

        with col1:
            # Market share pie chart
            pie_chart = MedisVisualizationUtils.create_market_share_pie(competitive)
            st.plotly_chart(pie_chart, use_container_width=True)

        with col2:
            # Competitor bar chart
            bar_chart = MedisVisualizationUtils.create_competitor_bar_chart(competitive)
            st.plotly_chart(bar_chart, use_container_width=True)

        # Market concentration and competitive metrics
        col3, col4, col5 = st.columns(3)

        with col3:
            hhi = competitive['market_concentration_hhi']
            concentration_level = "High" if hhi > 2500 else "Moderate" if hhi > 1500 else "Low"
            st.metric("Market Concentration", concentration_level, f"HHI: {hhi:.0f}")

        with col4:
            competitors = competitive['total_competitors']
            st.metric("Active Competitors", f"{competitors}", f"{competitors + 1} total labs")

        with col5:
            medis_rank = competitive['medis_rank']
            rank_change = f"#{medis_rank}"
            st.metric("MEDIS Rank", rank_change, "Market Leader" if medis_rank == 1 else "Challenger")

        # Competitive timeline
        st.subheader("📈 Market Share Evolution")
        timeline_chart = MedisVisualizationUtils.create_competitive_timeline(competitive)
        st.plotly_chart(timeline_chart, use_container_width=True)

        # Top competitors table
        st.subheader("🏭 Top Competitors Analysis")
        top_competitors = competitive['top_competitors']

        # Create comparison table
        competitor_data = []
        for lab, share in list(top_competitors.items())[:5]:
            competitor_data.append({
                'Laboratory': lab,
                'Market Share': ".1f",
                'Position': 'Market Leader' if lab == 'MEDIS' else 'Competitor'
            })

        competitor_df = pd.DataFrame(competitor_data)
        st.table(competitor_df)

    def _render_growth_analysis(self):
        """Render growth analysis section"""
        st.subheader("📈 Growth & Trend Analysis")

        growth = self.analysis_results['growth_analysis']
        medis_growth = growth['medis_growth']

        # Growth metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Starting Sales", f"{medis_growth['starting_sales']:,.0f}", "boxes/month")

        with col2:
            st.metric("Current Sales", f"{medis_growth['current_sales']:,.0f}", "boxes/month")

        with col3:
            st.metric("Peak Sales", f"{medis_growth['peak_sales']:,.0f}", medis_growth['peak_date'])

        with col4:
            st.metric("Total Growth", f"{medis_growth['total_growth']:+.1f}%", f"Avg: {medis_growth['avg_growth_per_year']:+.1f}%/year")

        # Growth trend chart
        st.subheader("📊 MEDIS Sales Growth Trend")
        growth_chart = MedisVisualizationUtils.create_growth_trend_chart(self.analysis_engine)
        st.plotly_chart(growth_chart, use_container_width=True)

        # Growth insights
        st.subheader("💡 Growth Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"🚀 **Growth Achievement**\n\nMEDIS achieved **{medis_growth['total_growth']:+.1f}%** total growth over {self.analysis_results['key_metrics']['months_active']} months")

        with col2:
            market_growth = growth['overall_growth']['yearly_growth']
            growth_vs_market = medis_growth['total_growth'] - market_growth
            if growth_vs_market > 0:
                st.success(f"📈 **Outperforming Market**\n\nMEDIS grew **{growth_vs_market:+.1f}%** more than market average ({market_growth:+.1f}%)")
            else:
                st.warning(f"📊 **Market Performance**\n\nMEDIS grew **{growth_vs_market:+.1f}%** vs market average ({market_growth:+.1f}%)")

        # Segment growth analysis
        st.subheader("💊 Growth by Market Segment")

        medis_segment_growth = growth['medis_segment_growth']
        segment_data = []
        for segment, sales in medis_segment_growth.items():
            segment_data.append({
                'Segment': segment,
                'MEDIS Sales': ".0f",
                'Performance': 'Strong' if sales > 100000 else 'Moderate' if sales > 50000 else 'Low'
            })

        segment_df = pd.DataFrame(segment_data).sort_values('MEDIS Sales', ascending=False)
        st.table(segment_df)

    def _render_seasonal_analysis(self):
        """Render seasonal analysis section"""
        st.subheader("📅 Seasonal Pattern Analysis")

        seasonal = self.analysis_results['seasonal_analysis']

        # Seasonal metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            market_peak = seasonal['peak_season']['market']
            st.metric("Market Peak Season", market_peak, "Highest demand")

        with col2:
            medis_peak = seasonal['peak_season']['medis']
            peak_match = "✓" if market_peak == medis_peak else "✗"
            st.metric("MEDIS Peak Season", f"{medis_peak} {peak_match}", "vs market")

        with col3:
            market_strength = seasonal['market_seasonal_strength']
            st.metric("Market Seasonality", f"{market_strength:.1f}%", "Sales variation")

        with col4:
            medis_strength = seasonal['medis_seasonal_strength']
            strength_diff = medis_strength - market_strength
            st.metric("MEDIS Seasonality", f"{medis_strength:.1f}%", f"{strength_diff:+.1f}% vs market")

        # Seasonal comparison chart
        seasonal_chart = MedisVisualizationUtils.create_seasonal_pattern_chart(seasonal)
        st.plotly_chart(seasonal_chart, use_container_width=True)

        # Seasonal insights
        st.subheader("🔍 Seasonal Insights")

        if seasonal['medis_seasonal_strength'] > seasonal['market_seasonal_strength']:
            st.info("📈 **Stronger Seasonal Patterns**: MEDIS shows more pronounced seasonal behavior than the overall market")
        else:
            st.info("📊 **Market-Aligned Seasonality**: MEDIS follows general market seasonal trends")

        # Peak and low season analysis
        peak_diff = abs(seasonal['medis_seasonal_pattern'][seasonal['peak_season']['medis']] -
                       seasonal['market_seasonal_pattern'][seasonal['peak_season']['market']])

        if peak_diff > 5000:
            st.warning("⚠️ **Seasonal Mismatch**: MEDIS peak season significantly differs from market peak")
        else:
            st.success("✅ **Seasonal Alignment**: MEDIS peak season aligns well with market patterns")

    def _render_product_analysis(self):
        """Render product and market segment analysis"""
        st.subheader("💊 Product & Market Segment Analysis")

        products = self.analysis_results['product_analysis']

        # Product metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            total_products = len(products['product_performance'])
            st.metric("Total Products", total_products, f"MEDIS: {len(products['medis_products'])}")

        with col2:
            total_segments = len(products['segment_performance'])
            st.metric("Market Segments", total_segments, "Dosage categories")

        with col3:
            top_product = max(products['product_performance'].items(), key=lambda x: x[1])
            st.metric("Top Product", top_product[0], f"{top_product[1]:.1f}% market")

        # Segment performance chart
        st.subheader("📊 Market Segment Performance")
        segment_chart = MedisVisualizationUtils.create_segment_performance_chart(products)
        st.plotly_chart(segment_chart, use_container_width=True)

        # MEDIS segment performance table
        st.subheader("🏆 MEDIS Segment Performance")

        medis_segment_data = []
        for segment, share in products['medis_segment_share'].items():
            market_size = products['segment_performance'].get(segment, 0)
            performance_level = "Excellent" if share > 30 else "Good" if share > 20 else "Moderate" if share > 10 else "Low"

            medis_segment_data.append({
                'Segment': segment,
                'MEDIS Share': ".1f",
                'Market Size': ".1f",
                'Performance': performance_level
            })

        medis_segment_df = pd.DataFrame(medis_segment_data).sort_values('MEDIS Share', ascending=False)
        st.table(medis_segment_df)

        # Top products analysis
        st.subheader("⭐ Top Products Analysis")

        top_products = products['top_products']
        product_data = []
        for product, sales in list(top_products.items())[:5]:
            is_medis = product in products['medis_products']
            product_data.append({
                'Product': product,
                'Sales': ".0f",
                'Market Share': ".1f",
                'Owned by MEDIS': '✓' if is_medis else '✗'
            })

        product_df = pd.DataFrame(product_data)
        st.table(product_df)

    def _render_data_quality_section(self):
        """Render data quality and insights section"""
        st.subheader("🔍 Data Quality & Business Insights")

        # Data quality report
        data_quality = self.data_loader.validate_data_quality()
        MedisVisualizationUtils.create_data_quality_report(data_quality)

        # Business insights
        business_insights = self.analysis_results['business_insights']
        MedisVisualizationUtils.create_insights_cards(business_insights)

    def add_sidebar_filters(self):
        """Add sidebar filters for data exploration"""
        st.sidebar.header("🔍 Data Filters")

        # Date range filter
        if self.analysis_results:
            date_range = self.analysis_results['key_metrics']['date_range']
            start_date = st.sidebar.date_input(
                "Start Date",
                value=date_range[0],
                min_value=date_range[0],
                max_value=date_range[1]
            )

            end_date = st.sidebar.date_input(
                "End Date",
                value=date_range[1],
                min_value=date_range[0],
                max_value=date_range[1]
            )

            if start_date > end_date:
                st.sidebar.error("Start date must be before end date")

        # Laboratory filter
        if self.data_loader:
            laboratories = ['All'] + sorted(self.data_loader.get_data()['laboratoire'].unique())
            selected_lab = st.sidebar.selectbox("Filter by Laboratory", laboratories)

        # Market segment filter
        if self.data_loader:
            segments = ['All'] + sorted(self.data_loader.get_data()['SOUS_MARCHE'].unique())
            selected_segment = st.sidebar.selectbox("Filter by Segment", segments)

        # Package size filter
        if self.data_loader:
            packages = ['All'] + sorted(self.data_loader.get_data()['pack_std'].dropna().unique())
            selected_package = st.sidebar.selectbox("Filter by Package Size", packages)

        return {
            'start_date': start_date if 'start_date' in locals() else None,
            'end_date': end_date if 'end_date' in locals() else None,
            'laboratory': selected_lab if 'selected_lab' in locals() else 'All',
            'segment': selected_segment if 'selected_segment' in locals() else 'All',
            'package': selected_package if 'selected_package' in locals() else 'All'
        }
