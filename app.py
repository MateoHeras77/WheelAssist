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

def create_interactive_scenario_builder():
    """Create interactive scenario builder in sidebar"""
    
    st.sidebar.markdown('<div class="sidebar-header">🎛️ Custom Scenario Builder</div>', unsafe_allow_html=True)
    
    # Agent reallocation controls
    st.sidebar.markdown("**Morning Boost (6 AM - 12 PM)**")
    morning_agents = st.sidebar.slider(
        "Additional agents",
        min_value=0,
        max_value=10,
        value=4,
        step=1,
        key="morning_agents"
    )
    
    st.sidebar.markdown("**Evening Reduction (2 PM - 8 PM)**")
    evening_reduction = st.sidebar.slider(
        "Agents to remove",
        min_value=0,
        max_value=10,
        value=4,
        step=1,
        key="evening_reduction"
    )
    
    # Time range selectors
    st.sidebar.markdown("**Fine-tune Time Windows**")
    
    morning_start = st.sidebar.time_input(
        "Morning boost start",
        value=time(8, 0),
        key="morning_start"
    )
    
    morning_end = st.sidebar.time_input(
        "Morning boost end",
        value=time(12, 0),
        key="morning_end"
    )
    
    evening_start = st.sidebar.time_input(
        "Evening reduction start",
        value=time(16, 0),
        key="evening_start"
    )
    
    evening_end = st.sidebar.time_input(
        "Evening reduction end",
        value=time(20, 0),
        key="evening_end"
    )
    
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
    
    return custom_adjustments

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">♿ WheelAssist Optimization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Optimizing wheelchair assistance staffing at Toronto Pearson Terminal 1**")
    
    # Load data
    analysis_results = load_data()
    if analysis_results is None:
        st.stop()
    
    simulator = get_simulator()
    
    # Sidebar controls
    st.sidebar.markdown('<div class="sidebar-header">📊 Analysis Controls</div>', unsafe_allow_html=True)
    
    # Scenario selector
    scenario_names = [s['scenario_name'] for s in analysis_results['detailed_results']]
    selected_scenario = st.sidebar.selectbox(
        "Select Scenario",
        scenario_names,
        index=scenario_names.index(analysis_results['best_scenario']['scenario_name'])
    )
    
    # Custom scenario builder
    custom_adjustments = create_interactive_scenario_builder()
    
    use_custom = st.sidebar.checkbox("Use Custom Scenario", value=False)
    
    # Main content
    if use_custom:
        # Evaluate custom scenario
        custom_scenario = {
            'name': 'Custom',
            'description': f'Custom reallocation: +{st.session_state.morning_agents} AM, -{st.session_state.evening_reduction} PM',
            'adjustments': custom_adjustments
        }
        
        current_result = simulator.evaluate_scenario(custom_scenario)
        current_gaps = current_result['scenario_gaps']
        scenario_title = "Custom Scenario"
    else:
        # Use selected predefined scenario
        current_result = next(r for r in analysis_results['detailed_results'] if r['scenario_name'] == selected_scenario)
        current_gaps = current_result['scenario_gaps']
        scenario_title = selected_scenario
    
    # Key metrics row
    baseline_result = analysis_results['detailed_results'][0]  # Baseline is first
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        understaffing_change = current_result['baseline_understaffing'] - current_result['scenario_understaffing']
        improvement_pct = (understaffing_change / max(current_result['baseline_understaffing'], 1)) * 100
        
        st.metric(
            "Understaffing Reduction",
            f"{understaffing_change:.1f} pax-hrs",
            f"{improvement_pct:.1f}%"
        )
    
    with col2:
        st.metric(
            "Wait Time Reduction",
            f"{current_result['wait_time_reduction_min']:.1f} min",
            f"Per day average"
        )
    
    with col3:
        st.metric(
            "Fewer Complaints",
            f"{current_result['estimated_complaints_reduction']}",
            f"Daily estimate"
        )
    
    with col4:
        morning_improvement = current_result['morning_improvement']
        st.metric(
            "Morning Peak Fix",
            f"{morning_improvement:.1f} gap units",
            f"6 AM - 12 PM period"
        )
    
    # Visualizations
    st.markdown("---")
    
    # Demand vs Capacity chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Current Scenario: Demand vs Capacity")
        demand_chart = create_demand_vs_capacity_chart(current_gaps, f"Demand vs Capacity - {scenario_title}")
        st.plotly_chart(demand_chart, use_container_width=True)
    
    with col2:
        st.subheader("📊 Baseline vs Current Comparison")
        baseline_gaps = baseline_result['baseline_gaps']
        
        # Create comparison chart
        comparison_fig = go.Figure()
        
        comparison_fig.add_trace(go.Scatter(
            x=baseline_gaps['hour'],
            y=baseline_gaps['gap'],
            mode='lines+markers',
            name='Baseline Gap',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        comparison_fig.add_trace(go.Scatter(
            x=current_gaps['hour'],
            y=current_gaps['gap'],
            mode='lines+markers',
            name=f'{scenario_title} Gap',
            line=dict(color='#1f77b4', width=3)
        ))
        
        comparison_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        comparison_fig.update_layout(
            title="Staffing Gap Comparison",
            xaxis_title="Hour of Day",
            yaxis_title="Gap (Demand - Capacity)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Scenario performance and heatmap
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 All Scenarios Performance")
        improvement_chart = create_improvement_chart(analysis_results['summary'])
        st.plotly_chart(improvement_chart, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Gap Heatmap by Scenario")
        heatmap = create_gap_heatmap(simulator, analysis_results['scenarios'])
        st.plotly_chart(heatmap, use_container_width=True)
    
    # Business recommendations
    st.markdown("---")
    st.subheader("💼 Business Recommendations")
    
    if use_custom:
        description = custom_scenario['description']
    else:
        description = current_result['description']
    
    rec_col1, rec_col2 = st.columns([2, 1])
    
    with rec_col1:
        st.markdown(f"""
        **Recommended Action:** {description}
        
        **Expected Outcomes:**
        - 🎯 **{improvement_pct:.1f}% reduction** in understaffing during peak hours
        - ⏱️ **{current_result['wait_time_reduction_min']:.1f} minutes less** average wait time
        - 😊 **{current_result['estimated_complaints_reduction']} fewer** passenger complaints daily
        - 💰 **Better resource utilization** - reduce evening overstaffing
        
        **Implementation Priority:**
        1. Focus on morning peak hours (6 AM - 12 PM)
        2. Prioritize gates E75, E80, E85 (higher distance factors)
        3. Monitor wait times and adjust weekly
        """)
    
    with rec_col2:
        st.markdown("""
        **Key Success Metrics:**
        - Wait time < 15 minutes
        - Complaints < 3 per day
        - Zero delayed boardings
        - Balanced capacity utilization
        """)
    
    # Detailed data tables
    with st.expander("📋 View Detailed Hourly Data"):
        
        tab1, tab2, tab3 = st.tabs(["Current Scenario", "Scenario Comparison", "Raw Data"])
        
        with tab1:
            st.dataframe(
                current_gaps[['hour', 'demand', 'capacity', 'gap', 'understaffed', 'overstaffed']],
                use_container_width=True
            )
        
        with tab2:
            st.dataframe(
                analysis_results['summary'],
                use_container_width=True
            )
        
        with tab3:
            # Show sample of original data
            st.markdown("**Flight Demand Sample:**")
            sample_flights = pd.read_csv('data/synthetic/flight_demand.csv').head(10)
            st.dataframe(sample_flights, use_container_width=True)

if __name__ == "__main__":
    main()
