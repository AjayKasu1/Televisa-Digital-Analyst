# TelevisaUnivision Digital Analyst - Streamlit Portfolio App
# Author: [Your Name]
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Page Config
st.set_page_config(layout="wide", page_title="Televisa Digital Analyst Dashboard", page_icon="📊")

# Title & Description
st.title("📊 TelevisaUnivision Digital Analyst Portfolio")
st.markdown("""
This application demonstrates **Automated Reporting**, **Anomaly Detection**, and **AI-Driven Insights** 
for Digital Ad Campaigns. It simulates a real-time ad server export of 100 campaigns.
""")

# ------------------------------------------------------------------------------
# 1. DATA GENERATION
# ------------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(num_campaigns=100):
    np.random.seed(42) 
    platforms = ['Linear', 'Digital', 'Streaming']
    clients = [f'Client_{i}' for i in range(1, 21)]
    data = []
    today = datetime.now()
    
    for i in range(num_campaigns):
        start_date = today - timedelta(days=random.randint(1, 60))
        duration = random.randint(14, 90)
        end_date = start_date + timedelta(days=duration)
        budget = round(np.random.uniform(5000, 50000), 2)
        cpm = np.random.uniform(10, 25)
        impressions_goal = int((budget / cpm) * 1000)
        
        total_days = (end_date - start_date).days
        days_elapsed = (today - start_date).days
        days_elapsed = max(0, min(days_elapsed, total_days))
        
        if days_elapsed == 0:
            impressions_delivered = 0
        else:
            expected_delivery_pct = days_elapsed / total_days
            variance = np.random.normal(1.0, 0.2) 
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

# ------------------------------------------------------------------------------
# 2. LOGIC & METRICS
# ------------------------------------------------------------------------------
def calculate_pacing(row):
    if row['Days_Elapsed'] == 0 or row['Impressions_Goal'] == 0:
        return 0.0
    goal_pacing_pct = row['Days_Elapsed'] / row['Total_Days']
    actual_pacing_pct = row['Impressions_Delivered'] / row['Impressions_Goal']
    if goal_pacing_pct == 0: return 0.0
    return actual_pacing_pct / goal_pacing_pct

df['Pacing_Index'] = df.apply(calculate_pacing, axis=1)
df['Pacing_%'] = (df['Pacing_Index'] * 100).round(2)
df['Days_Remaining'] = df['Total_Days'] - df['Days_Elapsed']
df['Time_Elapsed_%'] = (df['Days_Elapsed'] / df['Total_Days']) * 100
df['Budget_Spent_%'] = (df['Impressions_Delivered'] / df['Impressions_Goal']) * 100
df['eCPM'] = (df['Budget'] / df['Impressions_Delivered']) * 1000
df['eCPM'] = df['eCPM'].replace([np.inf, -np.inf], 0).fillna(0)

# Status Flagging
df['Status'] = 'On Track'
mask_live = df['Days_Elapsed'] > 5
df.loc[mask_live & (df['Pacing_%'] < 90), 'Status'] = 'Under-Pacing (At Risk)'
df.loc[mask_live & (df['Pacing_%'] > 110), 'Status'] = 'Over-Pacing (Overspend)'

at_risk_df = df[df['Status'] != 'On Track'].copy()

# ------------------------------------------------------------------------------
# 3. SIDEBAR & KPI metrics
# ------------------------------------------------------------------------------
st.sidebar.header("Filter Options")
selected_platform = st.sidebar.multiselect("Select Platform", df['Platform'].unique(), default=df['Platform'].unique())

filtered_df = df[df['Platform'].isin(selected_platform)]
at_risk_filtered = at_risk_df[at_risk_df['Platform'].isin(selected_platform)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Campaigns", len(filtered_df))
col1.metric("Total Budget", f"${filtered_df['Budget'].sum():,.0f}")

count_at_risk = len(at_risk_filtered)
col2.metric("At-Risk Campaigns", count_at_risk, delta=-count_at_risk, delta_color="inverse")
budget_at_risk = at_risk_filtered['Budget'].sum()
col2.metric("Revenue at Risk", f"${budget_at_risk:,.0f}", delta=-budget_at_risk, delta_color="inverse")

col3.metric("Avg eCPM", f"${filtered_df['eCPM'].mean():.2f}")


# ------------------------------------------------------------------------------
# 4. VISUALIZATIONS
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚀 Executive Dashboard")

tab1, tab2 = st.tabs(["Deep Dive Charts", "AI Insights Agent"])

with tab1:
    c1, c2 = st.columns(2)
    
    # Chart 1: Revenue at Risk
    risk_summary = at_risk_filtered.groupby(['Platform', 'Status'])['Budget'].sum().reset_index()
    fig1 = px.bar(
        risk_summary, x='Platform', y='Budget', color='Status',
        title='<b>1. Revenue at Risk</b> (Total Budget of At-Risk Campaigns)',
        color_discrete_map={'Under-Pacing (At Risk)': '#EF553B', 'Over-Pacing (Overspend)': '#FFA15A', 'On Track': '#00CC96'},
        text_auto='$.2s', template='plotly_white'
    )
    c1.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Pacing Heatmap
    fig2 = px.density_heatmap(
        filtered_df, x='Days_Remaining', y='Platform', z='Pacing_%', histfunc='avg',
        title='<b>2. Deadline Urgency Heatmap</b> (Avg Pacing %)',
        labels={'Days_Remaining': 'Days Remaining', 'Pacing_%': 'Avg Pacing %'},
        color_continuous_scale='YlOrRd', nbinsx=20, text_auto=True, template='plotly_white'
    )
    c2.plotly_chart(fig2, use_container_width=True)
    
    c3, c4 = st.columns(2)
    
    # Chart 3: Scatter
    fig3 = px.scatter(
        filtered_df, x='Time_Elapsed_%', y='Budget_Spent_%', size='Budget', color='Status',
        hover_name='Client_Name', title='<b>3. "High Stakes" Campaign Health</b> (Size=Budget)',
        color_discrete_map={'Under-Pacing (At Risk)': '#EF553B', 'Over-Pacing (Overspend)': '#FFA15A', 'On Track': '#00CC96'},
        template='plotly_white'
    )
    fig3.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="Gray", width=2, dash="dot"))
    c3.plotly_chart(fig3, use_container_width=True)
    
    # Chart 4: eCPM
    fig4 = px.box(
        filtered_df, x='Platform', y='eCPM', color='Platform',
        title='<b>4. eCPM Yield Efficiency</b>', points="all", template='plotly_white'
    )
    c4.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.markdown("### 🤖 AI Recommendation Agent")
    st.caption("Simulated AI logic providing operational recommendations for At-Risk campaigns.")
    
    def ai_insight_agent(row):
        pacing = row['Pacing_%']
        platform = row['Platform']
        if "Under-Pacing" in row['Status']:
            urgency = "HIGH" if pacing < 70 else "MEDIUM"
            return f"[{urgency}] Pacing {pacing}%. Increase bids on {platform} by 15%."
        elif "Over-Pacing" in row['Status']:
            return f"[WARNING] Pacing {pacing}% (Too Fast). Tighten frequency caps on {platform}."
        return "Monitor."
    
    if not at_risk_filtered.empty:
        at_risk_filtered['AI_Recommendation'] = at_risk_filtered.apply(ai_insight_agent, axis=1)
        st.dataframe(
            at_risk_filtered[['Campaign_ID', 'Client_Name', 'Platform', 'Pacing_%', 'Status', 'AI_Recommendation']]
            .style.applymap(lambda v: 'color: red;' if 'High' in str(v) else None)
        )
    else:
        st.success("✅ No campaigns are currently At-Risk!")

# ------------------------------------------------------------------------------
# 5. DATASET DOWNLOAD
# ------------------------------------------------------------------------------
st.markdown("---")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Full Campaign Data (CSV)", data=csv, file_name="televisa_campaign_data.csv", mime="text/csv")
