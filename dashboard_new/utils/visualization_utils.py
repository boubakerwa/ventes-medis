"""
MEDIS Pharmaceutical Sales Visualization Utilities

This module provides reusable visualization functions for the MEDIS sales forecasting dashboard.
It includes chart creation functions optimized for Streamlit with consistent styling and interactivity.

Author: AI Assistant
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style defaults
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MedisVisualizationUtils:
    """
    Visualization utilities for MEDIS pharmaceutical sales dashboard.

    Provides standardized chart creation functions with:
    - Consistent styling and colors
    - Interactive Plotly charts for Streamlit
    - Business-focused visualizations
    - Performance-optimized rendering
    """

    # Color scheme
    COLORS = {
        'medis': '#1f77b4',
        'competitors': '#ff7f0e',
        'market': '#2ca02c',
        'accent': '#d62728',
        'neutral': '#7f7f7f',
        'positive': '#17becf',
        'warning': '#bcbd22'
    }

    # Diverse color palette for multiple competitors
    COMPETITOR_COLORS = [
        '#1f77b4',  # Blue (MEDIS - keep consistent)
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#aec7e8',  # Light Blue
        '#ffbb78',  # Light Orange
        '#98df8a',  # Light Green
        '#ff9896',  # Light Red
        '#c5b0d5',  # Light Purple
        '#c49c94',  # Light Brown
        '#f7b6d2',  # Light Pink
        '#c7c7c7',  # Light Gray
        '#dbdb8d',  # Light Olive
        '#9edae5'   # Light Cyan
    ]

    @staticmethod
    def create_metric_cards(metrics: Dict[str, Any]) -> None:
        """
        Create metric cards for dashboard overview.

        Args:
            metrics: Dictionary of metrics from analysis engine
        """
        # Define metric cards with formatting
        metric_definitions = [
            {
                'label': 'Total Sales',
                'value': f"{metrics['total_sales']:,.0f}",
                'unit': 'boxes',
                'icon': '📦',
                'delta': f"{metrics['total_growth']:+.1f}%"
            },
            {
                'label': 'MEDIS Market Share',
                'value': f"{metrics['medis_market_share']:.1f}",
                'unit': '%',
                'icon': '🏆',
                'delta': f"{metrics['medis_market_share']:.1f}% of market"
            },
            {
                'label': 'MEDIS Sales',
                'value': f"{metrics['medis_sales']:,.0f}",
                'unit': 'boxes',
                'icon': '💊',
                'delta': f"Peak: {metrics['peak_monthly_sales']:,.0f}"
            },
            {
                'label': 'Active Competitors',
                'value': f"{metrics['active_competitors']}",
                'unit': 'companies',
                'icon': '🏭',
                'delta': f"Market period: {metrics['months_active']} months"
            }
        ]

        # Create columns and display cards
        cols = st.columns(len(metric_definitions))
        for col, metric in zip(cols, metric_definitions):
            with col:
                st.metric(
                    label=f"{metric['icon']} {metric['label']}",
                    value=f"{metric['value']} {metric['unit']}",
                    delta=metric['delta']
                )

    @staticmethod
    def create_market_share_pie(competitive_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create market share pie chart with diverse colors.

        Args:
            competitive_analysis: Competitive analysis results

        Returns:
            Plotly figure object
        """
        market_share = competitive_analysis['market_share_by_lab']

        # Sort by market share for better visualization
        sorted_companies = sorted(market_share.items(), key=lambda x: x[1], reverse=True)

        # Get top 8 companies for the pie chart (to avoid too many slices)
        top_companies = sorted_companies[:8]
        labels = [company for company, _ in top_companies]
        values = [share for _, share in top_companies]

        # Assign diverse colors, ensuring MEDIS gets the first color (blue)
        colors = []
        color_idx = 0
        for company in labels:
            if company == 'MEDIS':
                colors.append(MedisVisualizationUtils.COMPETITOR_COLORS[0])  # Blue for MEDIS
            else:
                color_idx += 1
                colors.append(MedisVisualizationUtils.COMPETITOR_COLORS[color_idx % len(MedisVisualizationUtils.COMPETITOR_COLORS)])

        # If there are more companies, add an "Others" slice
        if len(sorted_companies) > 8:
            remaining_companies = sorted_companies[8:]
            others_value = sum(share for _, share in remaining_companies)
            if others_value > 0:
                labels.append('Others')
                values.append(others_value)
                colors.append(MedisVisualizationUtils.COLORS['neutral'])  # Gray for others

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}% of market<br>%{text}<extra></extra>',
            text=[f"{v:,.0f} boxes" for v in values]
        )])

        fig.update_layout(
            title='Market Share Distribution (Top 8 Companies)',
            showlegend=False,
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    @staticmethod
    def create_growth_trend_chart(analysis_engine) -> go.Figure:
        """
        Create MEDIS growth trend chart.

        Args:
            analysis_engine: Instance of MedisAnalysisEngine

        Returns:
            Plotly figure object
        """
        # Get MEDIS monthly sales
        medis_monthly = analysis_engine.data_loader.get_monthly_sales('MEDIS')

        fig = go.Figure()

        # Add main sales line
        fig.add_trace(go.Scatter(
            x=medis_monthly['date'],
            y=medis_monthly['sales'],
            mode='lines+markers',
            name='MEDIS Sales',
            line=dict(color=MedisVisualizationUtils.COLORS['medis'], width=3),
            marker=dict(size=4),
            hovertemplate='<b>%{x}</b><br>Sales: %{y:,.0f} boxes<extra></extra>'
        ))

        # Add trend line (moving average)
        window = 12  # 12-month moving average
        if len(medis_monthly) >= window:
            ma = medis_monthly['sales'].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=medis_monthly['date'],
                y=ma,
                mode='lines',
                name=f'{window}-Month Trend',
                line=dict(color=MedisVisualizationUtils.COLORS['accent'], width=2, dash='dash'),
                hovertemplate=f'{window}-Month MA: %{{y:,.0f}}<extra></extra>'
            ))

        fig.update_layout(
            title='MEDIS Sales Growth Trend',
            xaxis_title='Date',
            yaxis_title='Monthly Sales (boxes)',
            hovermode='x unified',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    @staticmethod
    def create_competitor_bar_chart(competitive_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create competitor market share bar chart with diverse colors.

        Args:
            competitive_analysis: Competitive analysis results

        Returns:
            Plotly figure object
        """
        market_share = competitive_analysis['market_share_by_lab']

        # Sort by market share for better visualization
        sorted_companies = sorted(market_share.items(), key=lambda x: x[1], reverse=True)

        # Get top 10 competitors
        top_companies = sorted_companies[:10]
        labels = [company for company, _ in top_companies]
        values = [share for _, share in top_companies]

        # Assign diverse colors, ensuring MEDIS gets blue
        colors = []
        color_idx = 0
        for company in labels:
            if company == 'MEDIS':
                colors.append(MedisVisualizationUtils.COMPETITOR_COLORS[0])  # Blue for MEDIS
            else:
                color_idx += 1
                colors.append(MedisVisualizationUtils.COMPETITOR_COLORS[color_idx % len(MedisVisualizationUtils.COMPETITOR_COLORS)])

        fig = go.Figure(data=[go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>%{x:.1f}% market share<extra></extra>'
        )])

        fig.update_layout(
            title='Top 10 Laboratories by Market Share',
            xaxis_title='Market Share (%)',
            yaxis_title='Pharmaceutical Laboratory',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            height=max(400, len(labels) * 30),  # Dynamic height
            showlegend=False
        )

        return fig

    @staticmethod
    def create_seasonal_pattern_chart(seasonal_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create seasonal pattern comparison chart.

        Args:
            seasonal_analysis: Seasonal analysis results

        Returns:
            Plotly figure object
        """
        medis_pattern = seasonal_analysis['medis_seasonal_pattern']
        market_pattern = seasonal_analysis['market_seasonal_pattern']

        months = list(medis_pattern.keys())
        medis_values = list(medis_pattern.values())
        market_values = list(market_pattern.values())

        fig = go.Figure()

        # Add market pattern
        fig.add_trace(go.Scatter(
            x=months,
            y=market_values,
            mode='lines+markers',
            name='Market Average',
            line=dict(color=MedisVisualizationUtils.COLORS['market'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>Market: %{y:,.0f} boxes<extra></extra>'
        ))

        # Add MEDIS pattern
        fig.add_trace(go.Scatter(
            x=months,
            y=medis_values,
            mode='lines+markers',
            name='MEDIS',
            line=dict(color=MedisVisualizationUtils.COLORS['medis'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>MEDIS: %{y:,.0f} boxes<extra></extra>'
        ))

        fig.update_layout(
            title='Seasonal Sales Patterns Comparison',
            xaxis_title='Month',
            yaxis_title='Average Monthly Sales (boxes)',
            hovermode='x unified',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    @staticmethod
    def create_segment_performance_chart(product_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create dosage segment performance chart.

        Args:
            product_analysis: Product analysis results

        Returns:
            Plotly figure object
        """
        segment_performance = product_analysis['segment_performance']
        medis_segment_share = product_analysis['medis_segment_share']

        segments = list(segment_performance.keys())
        market_shares = list(segment_performance.values())
        medis_shares = [medis_segment_share.get(seg, 0) for seg in segments]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add market segment sizes
        fig.add_trace(go.Bar(
            x=segments,
            y=market_shares,
            name='Market Size',
            marker_color=MedisVisualizationUtils.COLORS['market'],
            hovertemplate='<b>%{x}</b><br>Market Size: %{y:.1f}%<extra></extra>'
        ), secondary_y=False)

        # Add MEDIS share in segments
        fig.add_trace(go.Scatter(
            x=segments,
            y=medis_shares,
            mode='lines+markers',
            name='MEDIS Share',
            line=dict(color=MedisVisualizationUtils.COLORS['medis'], width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>MEDIS Share: %{y:.1f}%<extra></extra>'
        ), secondary_y=True)

        fig.update_layout(
            title='Market Segments Performance',
            xaxis_title='Dosage Category',
            hovermode='x unified',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        fig.update_yaxes(title_text="Market Size (%)", secondary_y=False)
        fig.update_yaxes(title_text="MEDIS Share (%)", secondary_y=True)

        return fig

    @staticmethod
    def create_competitive_timeline(competitive_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create competitive timeline showing market share evolution.

        Args:
            competitive_analysis: Competitive analysis results

        Returns:
            Plotly figure object
        """
        medis_trend = competitive_analysis['medis_trend']

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=medis_trend['date'],
            y=medis_trend['market_share'],
            mode='lines+markers',
            name='MEDIS Market Share',
            line=dict(color=MedisVisualizationUtils.COLORS['medis'], width=3),
            marker=dict(size=4),
            hovertemplate='<b>%{x}</b><br>MEDIS Share: %{y:.1f}%<extra></extra>'
        ))

        # Add trend line if enough data
        if len(medis_trend) > 12:
            # Calculate 6-month moving average
            ma = medis_trend['market_share'].rolling(window=6).mean()
            fig.add_trace(go.Scatter(
                x=medis_trend['date'],
                y=ma,
                mode='lines',
                name='6-Month Trend',
                line=dict(color=MedisVisualizationUtils.COLORS['accent'], width=2, dash='dash'),
                hovertemplate='Trend: %{y:.1f}%<extra></extra>'
            ))

        fig.update_layout(
            title='MEDIS Market Share Evolution Over Time',
            xaxis_title='Date',
            yaxis_title='Market Share (%)',
            hovermode='x unified',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    @staticmethod
    def create_insights_cards(business_insights: Dict[str, Any]) -> None:
        """
        Create business insights cards.

        Args:
            business_insights: Business insights from analysis engine
        """
        st.subheader("💡 Business Insights")

        # Display insights
        for insight in business_insights['insights'][:3]:  # Show top 3
            st.info(insight)

        # Recommendations in expandable section
        with st.expander("📋 Strategic Recommendations"):
            for rec in business_insights['recommendations']:
                st.write(f"• {rec}")

        # Risk factors in expandable section
        with st.expander("⚠️ Risk Factors"):
            for risk in business_insights['risk_factors']:
                st.write(f"• {risk}")

    @staticmethod
    def create_data_quality_report(data_quality: Dict[str, Any]) -> None:
        """
        Create data quality report.

        Args:
            data_quality: Data quality metrics
        """
        st.subheader("🔍 Data Quality Report")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", f"{data_quality['total_records']:,}")

        with col2:
            null_pct = sum(data_quality['null_counts'].values()) / data_quality['total_records'] * 100
            st.metric("Data Completeness", f"{100-null_pct:.1f}%")

        with col3:
            st.metric("Duplicate Records", data_quality['duplicate_records'])

        # Null values breakdown
        if any(data_quality['null_counts'].values()):
            with st.expander("📊 Null Values by Column"):
                null_df = pd.DataFrame(
                    data_quality['null_counts'].items(),
                    columns=['Column', 'Null Count']
                ).sort_values('Null Count', ascending=False)

                for _, row in null_df.iterrows():
                    if row['Null Count'] > 0:
                        pct = row['Null Count'] / data_quality['total_records'] * 100
                        st.write(f"• **{row['Column']}**: {row['Null Count']:,} ({pct:.1f}%)")
