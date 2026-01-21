# TelevisaUnivision Digital Analyst - Automated Reporting Project
# Author: [Your Name]
# Purpose: Generate synthetic campaign data, analyze pacing/risk, and visualize insights.

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------------------------
# 1. DATA GENERATION
# ------------------------------------------------------------------------------
# Simulating a real-time ad server export with synthetic data.

def generate_synthetic_data(num_campaigns=100):
    np.random.seed(42) 
    
    platforms = ['Linear', 'Digital', 'Streaming']
    clients = [f'Client_{i}' for i in range(1, 21)]
    
    data = []
    today = datetime.now()
    
    for i in range(num_campaigns):
        # Generate realistic campaign parameters
        start_date = today - timedelta(days=random.randint(1, 60))
        duration = random.randint(14, 90)
        end_date = start_date + timedelta(days=duration)
        budget = round(np.random.uniform(5000, 50000), 2)
        cpm = np.random.uniform(10, 25)
        impressions_goal = int((budget / cpm) * 1000)
        
        # Calculate delivery parameters
        total_days = (end_date - start_date).days
        days_elapsed = (today - start_date).days
        days_elapsed = max(0, min(days_elapsed, total_days))
        
        # Simulate 'Human' variance in delivery (some campaigns under/over perform)
        if days_elapsed == 0:
            impressions_delivered = 0
        else:
            expected_delivery_pct = days_elapsed / total_days
            variance = np.random.normal(1.0, 0.2) # Normal distribution around 1.0
            actual_delivery_pct = expected_delivery_pct * variance
            impressions_delivered = int(impressions_goal * actual_delivery_pct)
            
        data.append({
            'Campaign_ID': f'CMP-{1000+i}',
            'Client_Name': random.choice(clients),
            'Platform': random.choice(platforms),
            'Impressions_Goal': impressions_goal,
            'Impressions_Delivered': impressions_delivered,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Budget': budget,
            'Total_Days': total_days,
            'Days_Elapsed': days_elapsed
        })
        
    return pd.DataFrame(data)

df = generate_synthetic_data()
print("Data Generation Complete: 100 Campaigns Created.")

# ------------------------------------------------------------------------------
# 2. PACING LOGIC & BUSINESS RULES
# ------------------------------------------------------------------------------
# Calculating "Pacing %" to determine if campaigns are ahead or behind schedule.

def calculate_pacing(row):
    if row['Days_Elapsed'] == 0 or row['Impressions_Goal'] == 0:
        return 0.0
    
    goal_pacing_pct = row['Days_Elapsed'] / row['Total_Days']
    actual_pacing_pct = row['Impressions_Delivered'] / row['Impressions_Goal']
    
    if goal_pacing_pct == 0: return 0.0
    
    return actual_pacing_pct / goal_pacing_pct

df['Pacing_Index'] = df.apply(calculate_pacing, axis=1)
df['Pacing_%'] = (df['Pacing_Index'] * 100).round(2)

# New Metrics for Visualization
df['Days_Remaining'] = df['Total_Days'] - df['Days_Elapsed']
df['Time_Elapsed_%'] = (df['Days_Elapsed'] / df['Total_Days']) * 100
# Calculate Budget Spent (Assuming linear spend proportional to impressions)
df['Budget_Spent_%'] = (df['Impressions_Delivered'] / df['Impressions_Goal']) * 100
df['eCPM'] = (df['Budget'] / df['Impressions_Delivered']) * 1000
df['eCPM'] = df['eCPM'].replace([np.inf, -np.inf], 0).fillna(0) # Handle division by zero

# ------------------------------------------------------------------------------
# 3. ANOMALY DETECTION
# ------------------------------------------------------------------------------
# Flagging campaigns pacing < 90% (Under) or > 110% (Over)

df['Status'] = 'On Track'
mask_live = df['Days_Elapsed'] > 5
df.loc[mask_live & (df['Pacing_%'] < 90), 'Status'] = 'Under-Pacing (At Risk)'
df.loc[mask_live & (df['Pacing_%'] > 110), 'Status'] = 'Over-Pacing (Overspend)'

at_risk_df = df[df['Status'] != 'On Track'].copy()
print(f"Anomaly Detection Complete: {len(at_risk_df)} At-Risk Campaigns Identified.")

# ------------------------------------------------------------------------------
# 4. AI INSIGHT AGENT
# ------------------------------------------------------------------------------
# Simulated agent that reads campaign status and prescribes action.

def ai_insight_agent(row):
    pacing = row['Pacing_%']
    platform = row['Platform']
    
    if "Under-Pacing" in row['Status']:
        urgency = "HIGH" if pacing < 70 else "MEDIUM"
        return (f"[{urgency} PRIORITY] Pacing at {pacing}%. "
                f"Rec: Increase bid caps on {platform} by 15% and expand audience targeting. "
                "Check for creative fatigue.")
    elif "Over-Pacing" in row['Status']:
        return (f"[WARNING] Pacing at {pacing}% (Too Fast). "
                f"Rec: Tighten frequency caps on {platform} immediately to conserve budget "
                "for the remainder of the flight.")
    return "Monitor."

at_risk_df['AI_Recommendation'] = at_risk_df.apply(ai_insight_agent, axis=1)

print("\nAI AGENT GENERATED REPORT (First 5 Rows):")
print(at_risk_df[['Campaign_ID', 'Platform', 'Pacing_%', 'Status', 'AI_Recommendation']].head().to_string(index=False))

# ------------------------------------------------------------------------------
# 5. VISUALIZATIONS
# ------------------------------------------------------------------------------
print("\nGenerating Executive Dashboards...")

# Chart 1: Revenue at Risk (Original)
risk_summary = at_risk_df.groupby(['Platform', 'Status'])['Budget'].sum().reset_index()

fig1 = px.bar(
    risk_summary, 
    x='Platform', 
    y='Budget', 
    color='Status',
    title='<b>Chart 1: Revenue at Risk</b><br><i>Total Budget of At-Risk Campaigns</i>',
    labels={'Budget': 'Total Budget at Risk ($)'},
    color_discrete_map={
        'Under-Pacing (At Risk)': '#EF553B', 
        'Over-Pacing (Overspend)': '#FFA15A',
        'On Track': '#00CC96'
    },
    text_auto='$.2s',
    template='plotly_white'
)
# fig1.show() # Commented out for Dashboard Export

# Chart 2: Pacing Heatmap (Platform vs Days Remaining)
fig2 = px.density_heatmap(
    df, 
    x='Days_Remaining', 
    y='Platform', 
    z='Pacing_%', 
    histfunc='avg', 
    title='<b>Chart 2: Deadline Urgency Heatmap</b><br><i>Avg Pacing % by Platform & Days Remaining</i>',
    labels={'Days_Remaining': 'Days Remaining in Campaign', 'Pacing_%': 'Avg Pacing %'},
    color_continuous_scale='YlOrRd', 
    nbinsx=20,
    text_auto=True,
    template='plotly_white'
)
# fig2.show()

# Chart 3: "High Stakes" Scatter Plot (Time vs Budget Spent)
fig3 = px.scatter(
    df, 
    x='Time_Elapsed_%', 
    y='Budget_Spent_%', 
    size='Budget', 
    color='Status',
    hover_name='Client_Name',
    hover_data=['Campaign_ID', 'Pacing_%'],
    title='<b>Chart 3: "High Stakes" Campaign Health</b><br><i>Budget Spent vs Time Elapsed (Size = Budget)</i>',
    color_discrete_map={
        'Under-Pacing (At Risk)': '#EF553B', 
        'Over-Pacing (Overspend)': '#FFA15A',
        'On Track': '#00CC96'
    },
    template='plotly_white'
)
# Add a reference line for 1:1 pacing (Ideal)
fig3.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="Gray", width=2, dash="dot"))
# fig3.show()

# Chart 4: eCPM Efficiency Gauge (Boxplot)
fig4 = px.box(
    df, 
    x='Platform', 
    y='eCPM', 
    color='Platform',
    title='<b>Chart 4: eCPM Yield Efficiency</b><br><i>Distribution of Effective Cost per Mille by Platform</i>',
    points="all", 
    template='plotly_white'
)
# fig4.show()

# ------------------------------------------------------------------------------
# 6. EXPORT TO HTML DASHBOARD (Single View)
# ------------------------------------------------------------------------------

# dashboard_filename = "Televisa_Digital_Analyst_Dashboard.html"
# print(f"\nExporting Dashboard to {dashboard_filename}...")

# with open(dashboard_filename, 'w') as f:
#     f.write("<html><head><title>TelevisaUnivision Digital Analyst Dashboard</title></head><body>")
#     f.write("<h1 style='font-family:Arial; text-align:center;'>Digital Campaign Performance Dashboard</h1>")
#     f.write("<hr>")
#     f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write("<hr>")
#     f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write("<hr>")
#     f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write("<hr>")
#     f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write("</body></html>")

# try:
#     # Try to open automatically if running locally
#     import webbrowser
#     import os
#     webbrowser.open('file://' + os.path.realpath(dashboard_filename))
# except:
#     pass

# print(f"Visualization Suite Generated Successfully. Open '{dashboard_filename}' to view.")

# ------------------------------------------------------------------------------
# 7. EXPORT DATASET
# ------------------------------------------------------------------------------
import os

folder_name = "simulated_data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

csv_filename = f"{folder_name}/televisa_campaign_data.csv"
df.to_csv(csv_filename, index=False)
print(f"Raw Dataset exported to folder: '{folder_name}/{os.path.basename(csv_filename)}'")
