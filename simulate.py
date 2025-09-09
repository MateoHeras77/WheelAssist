"""
Simulation Engine for WheelAssist ACAP Optimization
Calculates capacity, identifies gaps, and tests reallocation scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WheelAssistSimulator:
    def __init__(self, data_dir="data/synthetic"):
        """Initialize simulator with data loading"""
        self.data_dir = data_dir
        self.load_data()
        self.base_capacity_per_agent_hour = 6  # Base passengers per agent per hour
        
    def load_data(self):
        """Load all datasets"""
        print("📂 Loading datasets...")
        
        self.flight_demand = pd.read_csv(f"{self.data_dir}/flight_demand.csv")
        self.staffing_schedule = pd.read_csv(f"{self.data_dir}/staffing_schedule.csv")
        self.gate_metadata = pd.read_csv(f"{self.data_dir}/gate_metadata.csv")
        self.service_points = pd.read_csv(f"{self.data_dir}/service_points.csv")
        self.ops_metrics = pd.read_csv(f"{self.data_dir}/ops_metrics.csv")
        
        # Convert time columns
        self.flight_demand['arrival_time'] = pd.to_datetime(self.flight_demand['arrival_time'], format='%H:%M').dt.time
        self.staffing_schedule['start_time'] = pd.to_datetime(self.staffing_schedule['start_time'], format='%H:%M').dt.time
        self.staffing_schedule['end_time'] = pd.to_datetime(self.staffing_schedule['end_time'], format='%H:%M').dt.time
        
        print(f"✅ Loaded {len(self.flight_demand)} flights, {len(self.staffing_schedule)} shifts")
        
    def calculate_hourly_demand(self, date=None):
        """Calculate ACAP demand by hour for a specific date or average"""
        
        if date:
            day_flights = self.flight_demand[self.flight_demand['date'] == date].copy()
        else:
            # Calculate average across all dates
            day_flights = self.flight_demand.copy()
        
        # Extract hour from arrival time
        day_flights['hour'] = day_flights['arrival_time'].apply(lambda x: x.hour)
        
        # Group by hour and sum ACAP passengers
        hourly_demand = day_flights.groupby('hour')['pax_acaps'].sum().reindex(range(24), fill_value=0)
        
        if not date:
            # If calculating average, divide by number of unique dates
            n_dates = len(self.flight_demand['date'].unique())
            hourly_demand = hourly_demand / n_dates
            
        return hourly_demand
    
    def calculate_hourly_capacity(self, date=None, agent_adjustments=None):
        """Calculate available capacity by hour"""
        
        if date:
            day_shifts = self.staffing_schedule[self.staffing_schedule['date'] == date].copy()
        else:
            # Use first date as template
            first_date = self.staffing_schedule['date'].iloc[0]
            day_shifts = self.staffing_schedule[self.staffing_schedule['date'] == first_date].copy()
        
        # Apply agent adjustments if provided
        if agent_adjustments:
            day_shifts = day_shifts.copy()
            for adj in agent_adjustments:
                # Find shifts that overlap with the adjustment time window
                shift_mask = (
                    (day_shifts['start_time'] <= adj['end_time']) & 
                    (day_shifts['end_time'] >= adj['start_time'])
                )
                day_shifts.loc[shift_mask, 'agents_assigned'] += adj['agent_change']
                # Ensure minimum of 2 agents per shift
                day_shifts.loc[shift_mask, 'agents_assigned'] = day_shifts.loc[shift_mask, 'agents_assigned'].clip(lower=2)
        
        # Calculate capacity for each hour
        hourly_capacity = pd.Series(0.0, index=range(24))
        
        # Get service type adjustment (use Gate as primary service point)
        gate_service = self.service_points[self.service_points['service_type'] == 'Gate']
        service_adjustment = gate_service['agents_needed_factor'].iloc[0] if not gate_service.empty else 0.75
        
        # Average gate factor (distance impact)
        avg_gate_factor = self.gate_metadata['gate_factor'].mean()
        
        for _, shift in day_shifts.iterrows():
            start_hour = shift['start_time'].hour
            end_hour = shift['end_time'].hour
            
            # Handle overnight shifts
            if end_hour <= start_hour:
                hours = list(range(start_hour, 24)) + list(range(0, end_hour))
            else:
                hours = list(range(start_hour, end_hour))
            
            # Calculate capacity per agent for this shift
            capacity_per_agent = (self.base_capacity_per_agent_hour * service_adjustment) / avg_gate_factor
            
            for hour in hours:
                hourly_capacity[hour] += shift['agents_assigned'] * capacity_per_agent
        
        return hourly_capacity
    
    def calculate_gaps(self, date=None, agent_adjustments=None):
        """Calculate staffing gaps (demand - capacity) by hour"""
        
        demand = self.calculate_hourly_demand(date)
        capacity = self.calculate_hourly_capacity(date, agent_adjustments)
        
        gaps = demand - capacity
        
        # Create summary DataFrame
        gap_df = pd.DataFrame({
            'hour': range(24),
            'demand': demand.values,
            'capacity': capacity.values,
            'gap': gaps.values,
            'understaffed': (gaps > 0),
            'overstaffed': (gaps < 0)
        })
        
        return gap_df
    
    def generate_reallocation_scenarios(self, max_agents_to_move=10):
        """Generate various agent reallocation scenarios"""
        
        scenarios = []
        
        # Current baseline (no changes)
        scenarios.append({
            'name': 'Baseline (Current)',
            'description': 'Current staffing schedule',
            'adjustments': []
        })
        
        # Move agents from evening (14:00-22:00) to morning (06:00-12:00)
        for agents_to_move in range(2, max_agents_to_move + 1, 2):
            scenarios.append({
                'name': f'Move {agents_to_move} PM→AM',
                'description': f'Move {agents_to_move} agents from 14:00-22:00 to 06:00-12:00',
                'adjustments': [
                    {
                        'start_time': datetime.strptime('06:00', '%H:%M').time(),
                        'end_time': datetime.strptime('12:00', '%H:%M').time(),
                        'agent_change': agents_to_move
                    },
                    {
                        'start_time': datetime.strptime('14:00', '%H:%M').time(),
                        'end_time': datetime.strptime('22:00', '%H:%M').time(),
                        'agent_change': -agents_to_move
                    }
                ]
            })
        
        # Alternative: Move fewer agents but in more targeted time slots
        scenarios.append({
            'name': 'Targeted Peak Support',
            'description': 'Move 4 agents from 16:00-20:00 to 08:00-12:00',
            'adjustments': [
                {
                    'start_time': datetime.strptime('08:00', '%H:%M').time(),
                    'end_time': datetime.strptime('12:00', '%H:%M').time(),
                    'agent_change': 4
                },
                {
                    'start_time': datetime.strptime('16:00', '%H:%M').time(),
                    'end_time': datetime.strptime('20:00', '%H:%M').time(),
                    'agent_change': -4
                }
            ]
        })
        
        return scenarios
    
    def evaluate_scenario(self, scenario, date=None):
        """Evaluate a specific reallocation scenario"""
        
        baseline_gaps = self.calculate_gaps(date)
        scenario_gaps = self.calculate_gaps(date, scenario['adjustments'])
        
        # Calculate key metrics
        baseline_understaffing = baseline_gaps[baseline_gaps['gap'] > 0]['gap'].sum()
        scenario_understaffing = scenario_gaps[scenario_gaps['gap'] > 0]['gap'].sum()
        
        baseline_overstaffing = abs(baseline_gaps[baseline_gaps['gap'] < 0]['gap'].sum())
        scenario_overstaffing = abs(scenario_gaps[scenario_gaps['gap'] < 0]['gap'].sum())
        
        # Peak hour performance (6 AM - 12 PM)
        morning_hours = range(6, 12)
        baseline_morning_gap = baseline_gaps[baseline_gaps['hour'].isin(morning_hours)]['gap'].sum()
        scenario_morning_gap = scenario_gaps[scenario_gaps['hour'].isin(morning_hours)]['gap'].sum()
        
        # Evening efficiency (2 PM - 10 PM)
        evening_hours = range(14, 22)
        baseline_evening_gap = baseline_gaps[baseline_gaps['hour'].isin(evening_hours)]['gap'].sum()
        scenario_evening_gap = scenario_gaps[scenario_gaps['hour'].isin(evening_hours)]['gap'].sum()
        
        # Calculate improvements
        understaffing_improvement = baseline_understaffing - scenario_understaffing
        overstaffing_reduction = baseline_overstaffing - scenario_overstaffing
        morning_improvement = baseline_morning_gap - scenario_morning_gap
        
        # Estimate operational impact
        wait_time_reduction = max(0, understaffing_improvement * 2.5)  # Assume 2.5 min reduction per gap unit
        complaints_reduction = max(0, int(understaffing_improvement * 0.3))  # Fewer complaints
        
        results = {
            'scenario_name': scenario['name'],
            'description': scenario['description'],
            'baseline_understaffing': round(baseline_understaffing, 1),
            'scenario_understaffing': round(scenario_understaffing, 1),
            'understaffing_improvement': round(understaffing_improvement, 1),
            'understaffing_improvement_pct': round((understaffing_improvement / max(baseline_understaffing, 1)) * 100, 1),
            'baseline_overstaffing': round(baseline_overstaffing, 1),
            'scenario_overstaffing': round(scenario_overstaffing, 1),
            'overstaffing_reduction': round(overstaffing_reduction, 1),
            'morning_improvement': round(morning_improvement, 1),
            'wait_time_reduction_min': round(wait_time_reduction, 1),
            'estimated_complaints_reduction': complaints_reduction,
            'baseline_gaps': baseline_gaps,
            'scenario_gaps': scenario_gaps
        }
        
        return results
    
    def run_full_analysis(self):
        """Run complete analysis with all scenarios"""
        
        print("🔄 Running WheelAssist optimization analysis...")
        
        # Generate scenarios
        scenarios = self.generate_reallocation_scenarios()
        
        # Evaluate all scenarios
        results = []
        for scenario in scenarios:
            print(f"📊 Evaluating: {scenario['name']}")
            result = self.evaluate_scenario(scenario)
            results.append(result)
        
        # Create summary DataFrame
        summary_cols = [
            'scenario_name', 'description', 'baseline_understaffing', 'scenario_understaffing',
            'understaffing_improvement', 'understaffing_improvement_pct', 'morning_improvement',
            'wait_time_reduction_min', 'estimated_complaints_reduction'
        ]
        
        summary_df = pd.DataFrame([{col: r[col] for col in summary_cols} for r in results])
        
        # Find best scenario
        best_scenario_idx = summary_df['understaffing_improvement'].idxmax()
        best_scenario = results[best_scenario_idx]
        
        print("\n✅ Analysis complete!")
        print(f"🏆 Best scenario: {best_scenario['scenario_name']}")
        print(f"   📉 Understaffing reduction: {best_scenario['understaffing_improvement']} passenger-hours ({best_scenario['understaffing_improvement_pct']}%)")
        print(f"   ⏰ Wait time reduction: {best_scenario['wait_time_reduction_min']} minutes")
        print(f"   📝 Fewer complaints: {best_scenario['estimated_complaints_reduction']}")
        
        return {
            'summary': summary_df,
            'detailed_results': results,
            'best_scenario': best_scenario,
            'scenarios': scenarios
        }
    
    def get_hourly_breakdown(self, scenario_name=None):
        """Get detailed hourly breakdown for visualization"""
        
        if scenario_name:
            scenarios = self.generate_reallocation_scenarios()
            scenario = next((s for s in scenarios if s['name'] == scenario_name), scenarios[0])
            gaps = self.calculate_gaps(agent_adjustments=scenario['adjustments'])
        else:
            gaps = self.calculate_gaps()
        
        return gaps

def main():
    """Run simulation analysis"""
    
    print("🚁 WheelAssist ACAP Optimization Simulator")
    print("=" * 50)
    
    # Initialize simulator
    sim = WheelAssistSimulator()
    
    # Run full analysis
    analysis_results = sim.run_full_analysis()
    
    # Display summary table
    print("\n📋 Scenario Comparison Summary:")
    print("=" * 80)
    summary = analysis_results['summary']
    print(summary[['scenario_name', 'understaffing_improvement', 'understaffing_improvement_pct', 
                   'wait_time_reduction_min', 'estimated_complaints_reduction']].to_string(index=False))
    
    # Show baseline vs best scenario hourly comparison
    best_scenario_name = analysis_results['best_scenario']['scenario_name']
    
    print(f"\n📊 Hourly Breakdown: Baseline vs {best_scenario_name}")
    print("=" * 80)
    
    baseline_gaps = sim.get_hourly_breakdown()
    best_gaps = sim.get_hourly_breakdown(best_scenario_name)
    
    comparison = pd.DataFrame({
        'Hour': range(24),
        'Baseline_Gap': baseline_gaps['gap'].round(1),
        'Optimized_Gap': best_gaps['gap'].round(1),
        'Improvement': (baseline_gaps['gap'] - best_gaps['gap']).round(1)
    })
    
    # Show key hours (6 AM - 10 PM)
    key_hours = comparison[(comparison['Hour'] >= 6) & (comparison['Hour'] <= 22)]
    print(key_hours.to_string(index=False))
    
    print(f"\n💡 Business Recommendation:")
    print(f"   {analysis_results['best_scenario']['description']}")
    print(f"   Expected {analysis_results['best_scenario']['understaffing_improvement_pct']}% improvement in service levels")
    print(f"   Reduce average wait times by {analysis_results['best_scenario']['wait_time_reduction_min']} minutes")
    
    # Save results for Streamlit app
    import pickle
    with open('data/synthetic/analysis_results.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)
    print(f"\n💾 Results saved to data/synthetic/analysis_results.pkl")

if __name__ == "__main__":
    main()
