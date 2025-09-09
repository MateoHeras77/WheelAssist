"""
Data Generation Script for WheelAssist ACAP Optimization
Generates synthetic but realistic data for Toronto Pearson T1 wheelchair assistance analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_flight_demand(start_date="2024-01-01", days=30):
    """Generate synthetic flight demand data with morning peaks"""
    
    flights = []
    airlines = ["AC", "WS", "TS", "F8", "UA", "DL", "AA", "LH", "BA"]
    gates = ["E70", "E72", "E75", "E78", "E80", "E82", "E85", "E88", "E90"]
    
    # Flight times with morning concentration (6-12) and evening reduction
    morning_hours = list(range(6, 12)) * 8  # More morning flights
    afternoon_hours = list(range(12, 18)) * 5
    evening_hours = list(range(18, 23)) * 3  # Fewer evening flights
    flight_hours = morning_hours + afternoon_hours + evening_hours
    
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    
    for date in date_range:
        # More flights on weekdays
        day_of_week = date.strftime('%A')
        flights_per_day = 45 if day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 30
        
        for flight_num in range(flights_per_day):
            airline = random.choice(airlines)
            flight_number = f"{airline}{random.randint(100, 999)}"
            
            # Select arrival time with bias toward morning
            arrival_hour = random.choice(flight_hours)
            arrival_minute = random.choice([0, 15, 30, 45])
            arrival_time = f"{arrival_hour:02d}:{arrival_minute:02d}"
            
            # Total passengers vary by time of day and flight
            if arrival_hour < 12:  # Morning flights tend to be fuller
                pax_total = random.randint(120, 300)
            else:
                pax_total = random.randint(80, 250)
            
            # ACAP passengers (wheelchair assistance) - typically 2-8% of total
            # Higher percentage in morning due to elderly passengers
            if arrival_hour < 12:
                acap_rate = random.uniform(0.04, 0.08)  # 4-8% in morning
            else:
                acap_rate = random.uniform(0.02, 0.05)  # 2-5% later
                
            pax_acaps = max(1, int(pax_total * acap_rate))
            
            gate_id = random.choice(gates)
            season = "Winter" if date.month in [12, 1, 2] else "Summer" if date.month in [6, 7, 8] else "Shoulder"
            
            flights.append({
                'flight_id': flight_number,
                'date': date.strftime('%Y-%m-%d'),
                'arrival_time': arrival_time,
                'pax_total': pax_total,
                'pax_acaps': pax_acaps,
                'terminal': 'T1',
                'gate_id': gate_id,
                'day_of_week': day_of_week,
                'season': season
            })
    
    return pd.DataFrame(flights)

def generate_staffing_schedule(start_date="2024-01-01", days=30):
    """Generate staffing schedule with evening overstaffing pattern"""
    
    shifts = []
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Define shift patterns - currently overstaffed in evening, understaffed in morning
    shift_patterns = [
        {'start_time': '06:00', 'end_time': '14:00', 'agents_assigned': 8},   # Morning understaffed
        {'start_time': '10:00', 'end_time': '18:00', 'agents_assigned': 12},  # Mid-day
        {'start_time': '14:00', 'end_time': '22:00', 'agents_assigned': 18},  # Evening overstaffed
        {'start_time': '18:00', 'end_time': '02:00', 'agents_assigned': 6},   # Night
    ]
    
    shift_id = 1
    for date in date_range:
        day_of_week = date.strftime('%A')
        
        # Weekend adjustment
        weekend_factor = 0.8 if day_of_week in ['Saturday', 'Sunday'] else 1.0
        
        for pattern in shift_patterns:
            agents = max(4, int(pattern['agents_assigned'] * weekend_factor))
            
            shifts.append({
                'shift_id': f"SHIFT_{shift_id:04d}",
                'date': date.strftime('%Y-%m-%d'),
                'start_time': pattern['start_time'],
                'end_time': pattern['end_time'],
                'agents_assigned': agents,
                'team': 'ACAP',
                'day_of_week': day_of_week
            })
            shift_id += 1
    
    return pd.DataFrame(shifts)

def generate_gate_metadata():
    """Generate gate metadata with distance and service factors"""
    
    gates_data = [
        {'gate_id': 'E70', 'cluster': 'E70s', 'distance_m': 150, 'gate_factor': 1.0},
        {'gate_id': 'E72', 'cluster': 'E70s', 'distance_m': 180, 'gate_factor': 1.1},
        {'gate_id': 'E75', 'cluster': 'E70s', 'distance_m': 220, 'gate_factor': 1.2},
        {'gate_id': 'E78', 'cluster': 'E70s', 'distance_m': 250, 'gate_factor': 1.3},
        {'gate_id': 'E80', 'cluster': 'E80s', 'distance_m': 300, 'gate_factor': 1.4},
        {'gate_id': 'E82', 'cluster': 'E80s', 'distance_m': 320, 'gate_factor': 1.5},
        {'gate_id': 'E85', 'cluster': 'E80s', 'distance_m': 380, 'gate_factor': 1.6},
        {'gate_id': 'E88', 'cluster': 'E80s', 'distance_m': 420, 'gate_factor': 1.7},
        {'gate_id': 'E90', 'cluster': 'E90s', 'distance_m': 480, 'gate_factor': 1.8},
    ]
    
    return pd.DataFrame(gates_data)

def generate_service_points():
    """Generate service points with time and agent factors"""
    
    service_data = [
        {'service_type': 'Counter', 'avg_time_per_passenger_min': 3.0, 'agents_needed_factor': 1.0},
        {'service_type': 'Security', 'avg_time_per_passenger_min': 5.0, 'agents_needed_factor': 0.8},
        {'service_type': 'WaitingArea', 'avg_time_per_passenger_min': 2.0, 'agents_needed_factor': 0.9},
        {'service_type': 'Gate', 'avg_time_per_passenger_min': 8.0, 'agents_needed_factor': 0.75},
    ]
    
    return pd.DataFrame(service_data)

def generate_ops_metrics(start_date="2024-01-01", days=30):
    """Generate operational metrics showing current performance issues"""
    
    metrics = []
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    
    for date in date_range:
        day_of_week = date.strftime('%A')
        
        # Morning understaffing leads to more delays and complaints
        # Evening overstaffing leads to overtime
        if day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            delayed_boardings = random.randint(8, 15)  # More delays due to understaffing
            avg_wait_time = random.uniform(25, 45)     # Longer waits in morning
            complaints = random.randint(3, 8)          # More complaints
            agents_absent = random.randint(1, 3)
            overtime_hours = random.uniform(15, 30)    # Overtime from inefficient scheduling
        else:
            delayed_boardings = random.randint(3, 8)
            avg_wait_time = random.uniform(15, 30)
            complaints = random.randint(1, 4)
            agents_absent = random.randint(0, 2)
            overtime_hours = random.uniform(8, 20)
        
        metrics.append({
            'date': date.strftime('%Y-%m-%d'),
            'delayed_boardings': delayed_boardings,
            'avg_wait_time_min': round(avg_wait_time, 1),
            'complaints': complaints,
            'agents_absent': agents_absent,
            'overtime_hours': round(overtime_hours, 1),
            'day_of_week': day_of_week
        })
    
    return pd.DataFrame(metrics)

def main():
    """Generate all datasets and save to CSV files"""
    
    print("🛫 Generating WheelAssist synthetic data...")
    
    # Generate datasets
    print("📊 Generating flight demand data...")
    flight_demand = generate_flight_demand()
    
    print("👥 Generating staffing schedule...")
    staffing_schedule = generate_staffing_schedule()
    
    print("🚪 Generating gate metadata...")
    gate_metadata = generate_gate_metadata()
    
    print("🏢 Generating service points...")
    service_points = generate_service_points()
    
    print("📈 Generating operational metrics...")
    ops_metrics = generate_ops_metrics()
    
    # Save to CSV files
    output_dir = "data/synthetic"
    
    flight_demand.to_csv(f"{output_dir}/flight_demand.csv", index=False)
    staffing_schedule.to_csv(f"{output_dir}/staffing_schedule.csv", index=False)
    gate_metadata.to_csv(f"{output_dir}/gate_metadata.csv", index=False)
    service_points.to_csv(f"{output_dir}/service_points.csv", index=False)
    ops_metrics.to_csv(f"{output_dir}/ops_metrics.csv", index=False)
    
    print("\n✅ Data generation complete!")
    print(f"📁 Files saved to {output_dir}/")
    print(f"   - flight_demand.csv: {len(flight_demand)} flights")
    print(f"   - staffing_schedule.csv: {len(staffing_schedule)} shifts")
    print(f"   - gate_metadata.csv: {len(gate_metadata)} gates")
    print(f"   - service_points.csv: {len(service_points)} service types")
    print(f"   - ops_metrics.csv: {len(ops_metrics)} daily metrics")
    
    # Show sample data
    print("\n📋 Sample flight demand (first 5 rows):")
    print(flight_demand.head())
    
    print("\n👥 Sample staffing schedule (first 5 rows):")
    print(staffing_schedule.head())

if __name__ == "__main__":
    main()
