# ACAPs Manpower Optimization — Streamlit Project Guide

## Goal
Create a simple but realistic demo showing how understaffing in the morning and overstaffing in the evening affect wheelchair assistance (ACAPs) at Toronto Pearson T1, and how rebalancing agents can solve the problem.

---

## Variables (Core Datasets)

### `flight_demand`
- `flight_id` — flight code (e.g., AV255).  
- `date` — flight date.  
- `arrival_time` — scheduled arrival (hh:mm).  
- `pax_total` — total passengers.  
- `pax_acaps` — passengers requesting wheelchair assistance.  
- `terminal` — always T1 (keep flexible for T3).  
- `gate_id` — gate (e.g., E70, E75, E85).  
- `day_of_week` — Mon–Sun.  
- `season` — Summer, Winter, etc.

### `staffing_schedule`
- `shift_id` — unique shift identifier.  
- `date` — shift date.  
- `start_time` — shift start (hh:mm).  
- `end_time` — shift end (hh:mm).  
- `agents_assigned` — number of agents.  
- `team` — e.g., ACAP.  
- `day_of_week` — Mon–Sun.

### `gate_metadata`
- `gate_id` — gate code.  
- `cluster` — grouping (E70s, E80s).  
- `distance_m` — approximate walking distance.  
- `gate_factor` — adjustment (>1 means slower service due to distance).

### `service_points`
- `service_type` — Counter, Security, WaitingArea, Gate.  
- `avg_time_per_passenger_min` — average handling time.  
- `agents_needed_factor` — multiplier to adjust manpower requirements.

### `ops_metrics`
- `date` — operational date.  
- `delayed_boardings` — number of delayed boardings.  
- `avg_wait_time_min` — average ACAP wait time.  
- `complaints` — number of passenger complaints.  
- `agents_absent` — staff absent.  
- `overtime_hours` — overtime worked.

---

## Models (Analytical Layer)

### 1. Demand Modeling
- **Negative Binomial Regression** (preferred over Poisson if variance > mean).  
- Predictors:  
  - Hour of the day (`C(hour)`)  
  - Day of the week (`C(day_of_week)`)  
  - Gate cluster (`C(cluster)`)  
- Target: `pax_acaps` (per hour).  

### 2. Capacity Modeling
- **Capacity per agent** is not constant:  
capacity = base_pax_per_agent_hour * service_type_adjustment / gate_factor
- Example baseline: 6 pax/agent-hour.  
- Adjustments: Counter (1.0), Security (0.8), WaitingArea (0.9), Gate (0.75).  

### 3. Gap Analysis
- Calculate understaffing/overstaffing per hour:  
gap = demand_acaps_hour
- (agents_hour * capacity)
- `gap > 0` → understaffing.  
- `gap < 0` → overstaffing.  

### 4. Simulation (What-If)
- Reallocate agents from evening (14:00–22:00) to morning (06:00–12:00).  
- Measure:
- Baseline understaffing (pax-hours).  
- Rebalanced understaffing.  
- Improvement (Δ).  

### 5. Optimization
- Simple grid search: test moving 0–10 agents.  
- Select option with maximum improvement and minimal disruption.  

---

## Files and Instructions

### 1. `generate_data.py`
- Create synthetic data with morning ACAP peaks and evening overstaffing.  
- Save CSVs into `data/synthetic/`.

### 2. `simulate.py`
- Apply capacity formula.  
- Run what-if scenarios (move agents AM ↔ PM).  
- Output metrics for baseline vs rebalanced.

### 3. `app.py`
- Streamlit UI.  
- Sliders for number of agents reallocated.  
- Charts:  
- Demand vs staffing (line chart).  
- Gap distribution (bars).  
- KPI metrics: baseline understaffing, rebalanced understaffing, improvement.

### 4. `requirements.txt`
- Libraries: pandas, numpy, statsmodels, matplotlib or plotly, streamlit.

### 5. `README.md`
- One-paragraph problem description.  
- Quickstart commands.  
- KPIs of success: wait time, understaffing, overtime.  
- Clear statement: solution = **better service + lower cost + actionable recommendations**.

---

## Deliverables to Present
- Synthetic datasets (with documented assumptions).  
- 2–3 clean charts:  
- Demand vs staffing by hour.  
- Heatmap hour × day.  
- Gap before vs after rebalancing.  
- A before/after table with KPIs.  
- A short business explanation: *“Move N agents from PM to AM, prioritize E75/E85, expect X% reduction in wait time and Y fewer complaints.”*
