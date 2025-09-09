"""
WheelAssist ACAP Optimization - Streamlit Dashboard
Interactive dashboard for wheelchair assistance staffing optimization at Toronto Pearson T1
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, time
import sys
import os

# Add current directory to path to import simulate module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simulate import WheelAssistSimulator

# Page configuration
st.set_page_config(
    page_title="WheelAssist Optimization",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .improvement-positive {
        color: #28a745;
        font-weight: bold;
    }
    .improvement-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load simulation data and results"""
    try:
        # Load pre-computed analysis results
        with open('data/synthetic/analysis_results.pkl', 'rb') as f:
            analysis_results = pickle.load(f)
        return analysis_results
    except FileNotFoundError:
        st.error("Analysis results not found. Please run simulate.py first.")
        return None

@st.cache_resource
def get_simulator():
    """Get simulator instance"""
    return WheelAssistSimulator()

def create_demand_vs_capacity_chart(gaps_df, title="Demand vs Capacity by Hour"):
    """Create interactive demand vs capacity visualization"""
    
    fig = go.Figure()
    
    # Add demand line
    fig.add_trace(go.Scatter(
        x=gaps_df['hour'],
        y=gaps_df['demand'],
        mode='lines+markers',
        name='ACAP Demand',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Hour %{x}:00</b><br>Demand: %{y:.1f} passengers<extra></extra>'
    ))
    
    # Add capacity line
    fig.add_trace(go.Scatter(
        x=gaps_df['hour'],
        y=gaps_df['capacity'],
        mode='lines+markers',
        name='Available Capacity',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Hour %{x}:00</b><br>Capacity: %{y:.1f} passengers<extra></extra>'
    ))
    
    # Add gap fill
    fig.add_trace(go.Scatter(
        x=gaps_df['hour'],
        y=gaps_df['demand'],
        fill='tonexty',
        mode='none',
        name='Gap (Demand - Capacity)',
        fillcolor='rgba(255, 0, 0, 0.2)',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5),
        xaxis_title="Hour of Day",
        yaxis_title="Passengers per Hour",
        hovermode='x unified',
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    # Format x-axis to show hours properly
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
        )
    )
    
    return fig

def create_gap_heatmap(simulator, scenarios):
    """Create heatmap showing gaps across different scenarios"""
    
    # Calculate gaps for each scenario
    scenario_data = []
    for scenario in scenarios:
        if scenario['name'] == 'Baseline (Current)':
            gaps = simulator.get_hourly_breakdown()
        else:
            gaps = simulator.get_hourly_breakdown(scenario['name'])
        
        scenario_data.append({
            'scenario': scenario['name'],
            'gaps': gaps['gap'].values
        })
    
    # Create matrix for heatmap
    gap_matrix = np.array([s['gaps'] for s in scenario_data])
    scenario_names = [s['scenario'] for s in scenario_data]
    hours = list(range(24))
    
    fig = go.Figure(data=go.Heatmap(
        z=gap_matrix,
        x=hours,
        y=scenario_names,
        colorscale='RdBu_r',
        zmid=0,
        hovertemplate='<b>%{y}</b><br>Hour %{x}:00<br>Gap: %{z:.1f}<extra></extra>',
        colorbar=dict(title="Gap (Demand - Capacity)")
    ))
    
    fig.update_layout(
        title="Staffing Gaps by Scenario and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Scenario",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_improvement_chart(summary_df):
    """Create bar chart showing improvement metrics"""
    
    # Filter out baseline
    improvement_data = summary_df[summary_df['scenario_name'] != 'Baseline (Current)'].copy()
    
    fig = go.Figure()
    
    # Add understaffing improvement bars
    fig.add_trace(go.Bar(
        name='Understaffing Reduction (%)',
        x=improvement_data['scenario_name'],
        y=improvement_data['understaffing_improvement_pct'],
        marker_color='#28a745',
        hovertemplate='<b>%{x}</b><br>Understaffing Reduction: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Scenario Performance Comparison",
        xaxis_title="Reallocation Scenario",
        yaxis_title="Improvement (%)",
        height=400,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig

def create_data_overview_tab(simulator):
    """Create comprehensive data overview tab"""
    
    st.header("📊 Current Situation Analysis")
    st.markdown("**Understanding the current ACAP staffing challenges at Toronto Pearson T1**")
    
    # Load raw datasets
    flight_data = pd.read_csv('data/synthetic/flight_demand.csv')
    staffing_data = pd.read_csv('data/synthetic/staffing_schedule.csv')
    gate_data = pd.read_csv('data/synthetic/gate_metadata.csv')
    service_data = pd.read_csv('data/synthetic/service_points.csv')
    ops_data = pd.read_csv('data/synthetic/ops_metrics.csv')
    
    # Add airline extraction from flight_id
    flight_data['airline'] = flight_data['flight_id'].str[:2]
    flight_data['date'] = pd.to_datetime(flight_data['date'])
    
    # Filters section
    st.subheader("🔍 Data Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Location & Terminal**")
        selected_terminal = st.selectbox("Terminal", ["All", "T1"], index=1)
        selected_gates = st.multiselect(
            "Gates",
            sorted(flight_data['gate_id'].unique()),
            default=sorted(flight_data['gate_id'].unique())
        )
    
    with col2:
        st.markdown("**Time Filters**")
        selected_days = st.multiselect(
            "Day of Week", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        selected_season = st.multiselect(
            "Season",
            sorted(flight_data['season'].unique()),
            default=sorted(flight_data['season'].unique())
        )
    
    with col3:
        st.markdown("**Date Range**")
        min_date = flight_data['date'].min().date()
        max_date = flight_data['date'].max().date()
        
        st.info(f"Available data: {min_date} to {max_date}")
        
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD",
            help="Select start and end dates for analysis"
        )
        
        # Handle single date selection
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        elif isinstance(date_range, tuple) and len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date = end_date = date_range
    
    with col4:
        st.markdown("**Airlines & Flights**")
        available_airlines = sorted(flight_data['airline'].unique())
        selected_airlines = st.multiselect(
            "Airlines",
            available_airlines,
            default=available_airlines,
            help="First 2 characters of flight ID (e.g., AA from AA448)"
        )
        
        # Flight ID filter (optional)
        flight_id_filter = st.text_input(
            "Flight ID Filter (optional)",
            placeholder="e.g., AA448, WS, 255",
            help="Enter full flight ID, airline code, or flight number"
        )
    
    # Apply filters
    filtered_flights = flight_data[
        (flight_data['day_of_week'].isin(selected_days)) &
        (flight_data['gate_id'].isin(selected_gates)) &
        (flight_data['season'].isin(selected_season)) &
        (flight_data['airline'].isin(selected_airlines)) &
        (flight_data['date'].dt.date >= start_date) &
        (flight_data['date'].dt.date <= end_date)
    ]
    
    # Apply flight ID filter if provided
    if flight_id_filter.strip():
        filter_text = flight_id_filter.strip().upper()
        filtered_flights = filtered_flights[
            filtered_flights['flight_id'].str.contains(filter_text, case=False, na=False)
        ]
    
    # Filter statistics
    total_flights = len(flight_data)
    filtered_count = len(filtered_flights)
    filter_percentage = (filtered_count / total_flights * 100) if total_flights > 0 else 0
    
    st.info(f"📊 **Filter Results**: Showing {filtered_count:,} flights out of {total_flights:,} total ({filter_percentage:.1f}%)")
    
    # Show filter summary
    if filtered_count > 0:
        unique_dates = len(filtered_flights['date'].dt.date.unique())
        unique_airlines = len(filtered_flights['airline'].unique())
        unique_gates = len(filtered_flights['gate_id'].unique())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Date Range", f"{unique_dates} days")
        with col2:
            st.metric("Airlines", f"{unique_airlines} carriers")
        with col3:
            st.metric("Gates", f"{unique_gates} gates")
        with col4:
            avg_acap_rate = (filtered_flights['pax_acaps'].sum() / filtered_flights['pax_total'].sum() * 100) if filtered_flights['pax_total'].sum() > 0 else 0
            st.metric("ACAP Rate", f"{avg_acap_rate:.1f}%")
    
    # Current situation metrics
    st.markdown("---")
    st.subheader("📈 Current Situation Metrics")
    
    if filtered_count > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_daily_flights = filtered_count / unique_dates if unique_dates > 0 else 0
            st.metric("Avg Daily Flights", f"{avg_daily_flights:.0f}")
        
        with col2:
            avg_acap_rate = (filtered_flights['pax_acaps'].sum() / filtered_flights['pax_total'].sum()) * 100 if filtered_flights['pax_total'].sum() > 0 else 0
            st.metric("ACAP Rate", f"{avg_acap_rate:.1f}%")
        
        with col3:
            total_agents = staffing_data['agents_assigned'].sum() / len(staffing_data['date'].unique())
            st.metric("Daily Agents", f"{total_agents:.0f}")
        
        with col4:
            avg_complaints = ops_data['complaints'].mean()
            st.metric("Avg Daily Complaints", f"{avg_complaints:.1f}")
    else:
        st.warning("⚠️ No flights match the selected filters. Please adjust your filter criteria.")
        return
    
    # Detailed data sections
    st.markdown("---")
    
    # Filter Impact Analysis
    with st.expander("🔍 Filter Impact Analysis", expanded=False):
        st.markdown("**How your filters affect the data analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Selected Filters Summary:**")
            filter_summary = {
                "Date Range": f"{start_date} to {end_date}",
                "Airlines": f"{len(selected_airlines)} selected: {', '.join(selected_airlines)}",
                "Gates": f"{len(selected_gates)} selected: {', '.join(sorted(selected_gates))}",
                "Days": f"{len(selected_days)} selected: {', '.join(selected_days[:3])}{'...' if len(selected_days) > 3 else ''}",
                "Seasons": f"{len(selected_season)} selected: {', '.join(selected_season)}",
                "Flight ID Filter": flight_id_filter if flight_id_filter.strip() else "None"
            }
            
            for key, value in filter_summary.items():
                st.text(f"• {key}: {value}")
        
        with col2:
            st.markdown("**Data Comparison:**")
            
            # Compare filtered vs total data
            comparison_data = {
                "Metric": ["Total Flights", "Avg ACAP per Flight", "Peak Hour Flights", "Airlines Represented"],
                "Full Dataset": [
                    len(flight_data),
                    f"{flight_data['pax_acaps'].mean():.1f}",
                    flight_data.groupby(pd.to_datetime(flight_data['arrival_time'], format='%H:%M').dt.hour)['flight_id'].count().max(),
                    len(flight_data['airline'].unique())
                ],
                "Filtered Data": [
                    len(filtered_flights),
                    f"{filtered_flights['pax_acaps'].mean():.1f}" if len(filtered_flights) > 0 else "0.0",
                    filtered_flights.groupby(pd.to_datetime(filtered_flights['arrival_time'], format='%H:%M').dt.hour)['flight_id'].count().max() if len(filtered_flights) > 0 else 0,
                    len(filtered_flights['airline'].unique()) if len(filtered_flights) > 0 else 0
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Show filter impact percentage
            if len(filtered_flights) > 0:
                coverage_pct = len(filtered_flights) / len(flight_data) * 100
                st.metric("Data Coverage", f"{coverage_pct:.1f}%", help="Percentage of total data included in current filters")
    
    # Flight demand analysis
    with st.expander("✈️ Flight Demand Analysis", expanded=False):
        st.markdown("**Hourly flight distribution and ACAP demand patterns**")
        
        # Create hourly demand chart
        filtered_flights['hour'] = pd.to_datetime(filtered_flights['arrival_time'], format='%H:%M').dt.hour
        hourly_stats = filtered_flights.groupby('hour').agg({
            'flight_id': 'count',
            'pax_total': 'sum',
            'pax_acaps': 'sum'
        }).reset_index()
        hourly_stats.columns = ['Hour', 'Flights', 'Total_Passengers', 'ACAP_Passengers']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=hourly_stats['Hour'],
                y=hourly_stats['Flights'],
                name='Number of Flights',
                marker_color='lightblue',
                yaxis='y'
            ))
            fig_hourly.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['ACAP_Passengers'],
                name='ACAP Passengers',
                line=dict(color='red', width=3),
                yaxis='y2'
            ))
            
            fig_hourly.update_layout(
                title="Hourly Flight Distribution vs ACAP Demand",
                xaxis_title="Hour of Day",
                yaxis=dict(title="Number of Flights", side="left"),
                yaxis2=dict(title="ACAP Passengers", side="right", overlaying="y"),
                height=400
            )
            
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Airline analysis
            airline_stats = filtered_flights.groupby('airline').agg({
                'flight_id': 'count',
                'pax_total': 'sum',
                'pax_acaps': 'sum'
            }).reset_index()
            airline_stats['acap_rate'] = (airline_stats['pax_acaps'] / airline_stats['pax_total'] * 100).round(1)
            airline_stats.columns = ['Airline', 'Flights', 'Total_Pax', 'ACAP_Pax', 'ACAP_Rate_%']
            
            fig_airline = px.bar(
                airline_stats.sort_values('ACAP_Rate_%', ascending=False),
                x='Airline',
                y='ACAP_Rate_%',
                title="ACAP Rate by Airline",
                color='ACAP_Rate_%',
                color_continuous_scale='Reds',
                labels={'ACAP_Rate_%': 'ACAP Rate (%)'}
            )
            fig_airline.update_layout(height=400)
            st.plotly_chart(fig_airline, use_container_width=True)
        
        # Airline summary table
        st.markdown("**Airline Performance Summary**")
        st.dataframe(
            airline_stats.sort_values('Flights', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Show detailed flight data
        st.markdown("**Detailed Flight Data (Sample)**")
        display_flights = filtered_flights[['flight_id', 'airline', 'date', 'arrival_time', 'pax_total', 'pax_acaps', 'gate_id', 'day_of_week']].head(20)
        st.dataframe(display_flights, use_container_width=True, hide_index=True)
    
    # Gate information
    with st.expander("🚪 Gate Information & Factors", expanded=False):
        st.markdown("**Gate locations, distances, and service complexity factors**")
        
        # Gate factor explanation
        st.info("""
        **Gate Factor Explanation:**
        - Gate factor represents the service complexity multiplier for each gate
        - Higher values = longer walking distances = slower service
        - Factor ranges from 1.0 (closest gates) to 1.8+ (farthest gates)
        - Used in capacity calculation: capacity = base_rate × service_adjustment ÷ gate_factor
        """)
        
        # Enhanced gate data display
        gate_display = gate_data.copy()
        gate_display['Service_Impact'] = gate_display['gate_factor'].apply(
            lambda x: 'Low Impact' if x <= 1.2 else 'Medium Impact' if x <= 1.5 else 'High Impact'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gate factor visualization
            fig_gates = px.bar(
                gate_display,
                x='gate_id',
                y='gate_factor',
                color='cluster',
                title="Gate Service Complexity Factors",
                labels={'gate_factor': 'Service Factor', 'gate_id': 'Gate ID'}
            )
            st.plotly_chart(fig_gates, use_container_width=True)
        
        with col2:
            # Distance vs factor scatter
            fig_scatter = px.scatter(
                gate_display,
                x='distance_m',
                y='gate_factor',
                color='cluster',
                size='gate_factor',
                title="Distance vs Service Factor",
                labels={'distance_m': 'Distance (meters)', 'gate_factor': 'Service Factor'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.dataframe(gate_display, use_container_width=True)
    
    # Service points information
    with st.expander("🏢 Service Points & Time Requirements", expanded=False):
        st.markdown("**Different service locations and their time/agent requirements**")
        
        # Service points explanation
        st.info("""
        **Service Points Explanation:**
        - **Counter**: Initial check-in and assistance request (3 min/passenger)
        - **Security**: Security checkpoint assistance (5 min/passenger, 0.8 efficiency)
        - **Waiting Area**: Pre-boarding assistance (2 min/passenger, 0.9 efficiency)
        - **Gate**: Final boarding assistance (8 min/passenger, 0.75 efficiency due to distance)
        """)
        
        # Enhanced service data
        service_display = service_data.copy()
        service_display['Efficiency_Rating'] = service_display['agents_needed_factor'].apply(
            lambda x: 'High' if x >= 0.9 else 'Medium' if x >= 0.8 else 'Low'
        )
        service_display['Daily_Capacity_per_Agent'] = (60 / service_display['avg_time_per_passenger_min']) * service_display['agents_needed_factor'] * 8  # 8-hour shift
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time requirements chart
            fig_time = px.bar(
                service_display,
                x='service_type',
                y='avg_time_per_passenger_min',
                title="Average Time per Passenger by Service Point",
                color='avg_time_per_passenger_min',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Efficiency factors chart
            fig_efficiency = px.bar(
                service_display,
                x='service_type',
                y='agents_needed_factor',
                title="Agent Efficiency by Service Point",
                color='agents_needed_factor',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        st.dataframe(service_display, use_container_width=True)
    
    # Operations metrics
    with st.expander("📊 Current Operations Metrics", expanded=False):
        st.markdown("**Daily operational performance showing current issues**")
        
        # Ops metrics by day of week
        ops_by_day = ops_data.groupby('day_of_week').agg({
            'delayed_boardings': 'mean',
            'avg_wait_time_min': 'mean',
            'complaints': 'mean',
            'agents_absent': 'mean',
            'overtime_hours': 'mean'
        }).round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Wait times and complaints
            fig_performance = go.Figure()
            fig_performance.add_trace(go.Bar(
                x=ops_by_day.index,
                y=ops_by_day['avg_wait_time_min'],
                name='Avg Wait Time (min)',
                marker_color='orange'
            ))
            fig_performance.add_trace(go.Scatter(
                x=ops_by_day.index,
                y=ops_by_day['complaints'],
                name='Daily Complaints',
                line=dict(color='red', width=3),
                yaxis='y2'
            ))
            
            fig_performance.update_layout(
                title="Wait Times & Complaints by Day",
                xaxis_title="Day of Week",
                yaxis=dict(title="Wait Time (minutes)"),
                yaxis2=dict(title="Complaints", side="right", overlaying="y"),
                height=400
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        
        with col2:
            # Operational issues
            fig_ops = go.Figure()
            fig_ops.add_trace(go.Bar(
                x=ops_by_day.index,
                y=ops_by_day['delayed_boardings'],
                name='Delayed Boardings',
                marker_color='red'
            ))
            fig_ops.add_trace(go.Bar(
                x=ops_by_day.index,
                y=ops_by_day['overtime_hours'],
                name='Overtime Hours',
                marker_color='purple'
            ))
            
            fig_ops.update_layout(
                title="Operational Issues by Day",
                xaxis_title="Day of Week",
                yaxis_title="Count/Hours",
                height=400
            )
            st.plotly_chart(fig_ops, use_container_width=True)
        
        st.markdown("**Detailed Operations Data**")
        st.dataframe(ops_data, use_container_width=True)
    
    # Mathematical formulas
    with st.expander("🧮 Mathematical Formulas & Calculations", expanded=False):
        st.markdown("**Key formulas used in the optimization analysis**")
        
        st.markdown("""
        ### Core Capacity Formula
        ```
        Hourly Capacity = Agents × Base Rate × Service Adjustment ÷ Gate Factor
        ```
        
        **Where:**
        - **Base Rate**: 6 passengers per agent per hour (industry standard)
        - **Service Adjustment**: Efficiency factor by service type (0.75 - 1.0)
        - **Gate Factor**: Distance/complexity multiplier (1.0 - 1.8+)
        
        ### Gap Analysis
        ```
        Staffing Gap = Hourly ACAP Demand - Hourly Capacity
        ```
        - **Positive Gap**: Understaffing (demand > capacity)
        - **Negative Gap**: Overstaffing (capacity > demand)
        
        ### Performance Metrics
        ```
        Wait Time Impact = Gap × 2.5 minutes per passenger
        Complaint Estimate = Gap × 0.3 complaints per passenger
        Service Level = (1 - Gap/Demand) × 100%
        ```
        
        ### Example Calculation
        **Morning Peak (8 AM) - Current Situation:**
        - ACAP Demand: 25 passengers
        - Agents Available: 8
        - Service Type: Gate (0.75 efficiency)
        - Average Gate Factor: 1.4
        
        ```
        Capacity = 8 × 6 × 0.75 ÷ 1.4 = 25.7 passengers
        Gap = 25 - 25.7 = -0.7 (slight overstaffing)
        ```
        
        **However, demand varies significantly:**
        - Peak demand can reach 40+ passengers in a single hour
        - Current staffing creates gaps of 15-20 passengers during peak times
        """)

def create_solutions_tab(simulator, analysis_results):
    """Create solutions and optimization tab"""
    
    st.header("💡 Optimization Solutions")
    st.markdown("**Data-driven staffing reallocation strategies and their expected impact**")
    
    # Scenario selector and custom builder
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scenario_names = [s['scenario_name'] for s in analysis_results['detailed_results']]
        selected_scenario = st.selectbox(
            "Select Optimization Scenario",
            scenario_names,
            index=scenario_names.index(analysis_results['best_scenario']['scenario_name'])
        )
    
    with col2:
        use_custom = st.checkbox("Use Custom Scenario", value=False)
    
    # Custom scenario builder
    if use_custom:
        st.markdown("### 🎛️ Custom Scenario Builder")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            morning_agents = st.slider("Morning Boost (+agents)", 0, 10, 4)
            morning_start = st.time_input("Start Time", value=time(8, 0))
        
        with col2:
            evening_reduction = st.slider("Evening Reduction (-agents)", 0, 10, 4)
            evening_start = st.time_input("Reduce From", value=time(16, 0))
        
        with col3:
            morning_end = st.time_input("End Time", value=time(12, 0))
        
        with col4:
            evening_end = st.time_input("Reduce Until", value=time(20, 0))
        
        # Build custom scenario
        custom_adjustments = []
        if morning_agents > 0:
            custom_adjustments.append({
                'start_time': morning_start,
                'end_time': morning_end,
                'agent_change': morning_agents
            })
        if evening_reduction > 0:
            custom_adjustments.append({
                'start_time': evening_start,
                'end_time': evening_end,
                'agent_change': -evening_reduction
            })
        
        custom_scenario = {
            'name': 'Custom',
            'description': f'Custom: +{morning_agents} agents {morning_start}-{morning_end}, -{evening_reduction} agents {evening_start}-{evening_end}',
            'adjustments': custom_adjustments
        }
        
        current_result = simulator.evaluate_scenario(custom_scenario)
        current_gaps = current_result['scenario_gaps']
        scenario_title = "Custom Scenario"
    else:
        current_result = next(r for r in analysis_results['detailed_results'] if r['scenario_name'] == selected_scenario)
        current_gaps = current_result['scenario_gaps']
        scenario_title = selected_scenario
    
    # Solution impact metrics
    baseline_result = analysis_results['detailed_results'][0]
    
    st.markdown("---")
    st.subheader(f"📊 Impact Analysis: {scenario_title}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        understaffing_change = current_result['baseline_understaffing'] - current_result['scenario_understaffing']
        improvement_pct = (understaffing_change / max(current_result['baseline_understaffing'], 1)) * 100
        
        st.metric(
            "Understaffing Reduction",
            f"{understaffing_change:.1f} pax-hrs",
            f"{improvement_pct:.1f}% improvement"
        )
    
    with col2:
        st.metric(
            "Wait Time Reduction",
            f"{current_result['wait_time_reduction_min']:.1f} min",
            "Daily average"
        )
    
    with col3:
        st.metric(
            "Fewer Complaints",
            f"{current_result['estimated_complaints_reduction']}",
            "Per day estimate"
        )
    
    with col4:
        morning_improvement = current_result['morning_improvement']
        st.metric(
            "Morning Peak Fix",
            f"{morning_improvement:.1f} gap units",
            "6 AM - 12 PM period"
        )
    
    # Visualization section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Demand vs Capacity Comparison")
        demand_chart = create_demand_vs_capacity_chart(current_gaps, f"Optimized: {scenario_title}")
        st.plotly_chart(demand_chart, use_container_width=True)
    
    with col2:
        st.subheader("📊 Before vs After Gaps")
        baseline_gaps = baseline_result['baseline_gaps']
        
        comparison_fig = go.Figure()
        comparison_fig.add_trace(go.Scatter(
            x=baseline_gaps['hour'],
            y=baseline_gaps['gap'],
            mode='lines+markers',
            name='Current (Baseline)',
            line=dict(color='red', width=3)
        ))
        comparison_fig.add_trace(go.Scatter(
            x=current_gaps['hour'],
            y=current_gaps['gap'],
            mode='lines+markers',
            name=f'Optimized ({scenario_title})',
            line=dict(color='green', width=3)
        ))
        comparison_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        comparison_fig.update_layout(
            title="Staffing Gap: Current vs Optimized",
            xaxis_title="Hour of Day",
            yaxis_title="Gap (Demand - Capacity)",
            height=400
        )
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # All scenarios performance
    st.markdown("---")
    st.subheader("🏆 All Scenarios Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        improvement_chart = create_improvement_chart(analysis_results['summary'])
        st.plotly_chart(improvement_chart, use_container_width=True)
    
    with col2:
        heatmap = create_gap_heatmap(simulator, analysis_results['scenarios'])
        st.plotly_chart(heatmap, use_container_width=True)
    
    # Business recommendations
    st.markdown("---")
    st.subheader("💼 Implementation Recommendations")
    
    if use_custom:
        description = custom_scenario['description']
    else:
        description = current_result['description']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### Recommended Action
        **{description}**
        
        ### Expected Outcomes
        - 🎯 **{improvement_pct:.1f}% reduction** in understaffing during peak hours
        - ⏱️ **{current_result['wait_time_reduction_min']:.1f} minutes** less average wait time daily
        - 😊 **{current_result['estimated_complaints_reduction']} fewer** passenger complaints per day
        - 💰 **Better resource utilization** - reduce evening overstaffing waste
        
        ### Implementation Steps
        1. **Week 1**: Notify staff of schedule changes, begin transition
        2. **Week 2**: Implement new morning shift allocations
        3. **Week 3**: Monitor performance metrics, adjust as needed
        4. **Week 4**: Full implementation and performance review
        
        ### Priority Focus Areas
        - **Gates E75, E80, E85**: Highest distance factors (1.2-1.6)
        - **Morning Peak (8-11 AM)**: Highest demand concentration
        - **Service Point Optimization**: Focus on gate assistance efficiency
        """)
    
    with col2:
        st.markdown("""
        ### Success Metrics
        **Targets to Monitor:**
        - Wait time < 15 minutes
        - Complaints < 3 per day
        - Zero delayed boardings
        - Balanced utilization
        
        ### Risk Mitigation
        - **Staff Training**: Ensure smooth transition
        - **Backup Coverage**: Maintain minimum staffing
        - **Performance Monitoring**: Daily metric tracking
        - **Flexibility**: Adjust based on results
        
        ### Cost-Benefit
        - **No additional hiring** required
        - **Reduced overtime** costs
        - **Higher satisfaction** scores
        - **Improved efficiency** metrics
        """)
    
    # Detailed scenario comparison table
    with st.expander("📋 Detailed Scenario Comparison", expanded=False):
        st.dataframe(analysis_results['summary'], use_container_width=True)
    
    # Hourly breakdown table
    with st.expander("⏰ Hourly Gap Analysis", expanded=False):
        hourly_comparison = pd.DataFrame({
            'Hour': range(24),
            'Current_Demand': baseline_gaps['demand'],
            'Current_Capacity': baseline_gaps['capacity'],
            'Current_Gap': baseline_gaps['gap'].round(1),
            'Optimized_Capacity': current_gaps['capacity'],
            'Optimized_Gap': current_gaps['gap'].round(1),
            'Improvement': (baseline_gaps['gap'] - current_gaps['gap']).round(1)
        })
        
        # Color coding for gaps
        def color_gaps(val):
            if val > 5:
                return 'background-color: #ffcccc'  # Light red for high understaffing
            elif val > 0:
                return 'background-color: #ffe6cc'  # Light orange for mild understaffing
            elif val < -10:
                return 'background-color: #ccccff'  # Light blue for high overstaffing
            else:
                return 'background-color: #ccffcc'  # Light green for balanced
        
        styled_df = hourly_comparison.style.applymap(color_gaps, subset=['Current_Gap', 'Optimized_Gap'])
        st.dataframe(styled_df, use_container_width=True)

def main():
    """Main Streamlit application with enhanced tabs"""
    
    # Header
    st.markdown('<h1 class="main-header">♿ WheelAssist Optimization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced wheelchair assistance staffing optimization for Toronto Pearson Terminal 1**")
    
    # Load data
    analysis_results = load_data()
    if analysis_results is None:
        st.stop()
    
    simulator = get_simulator()
    
    # Create main tabs
    tab1, tab2 = st.tabs(["📊 Current Situation Analysis", "💡 Optimization Solutions"])
    
    with tab1:
        create_data_overview_tab(simulator)
    
    with tab2:
        create_solutions_tab(simulator, analysis_results)
if __name__ == "__main__":
    main()
