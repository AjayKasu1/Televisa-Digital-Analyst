import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.io as pio

# --- PART 1: Data Generation ---
# Simulating a dataset of 100 digital ad campaigns
def generate_synthetic_data(num_campaigns=100):
    np.random.seed(42)  # For reproducibility
    
    platforms = ['Linear', 'Digital', 'Streaming']
    clients = [f'Client_{i}' for i in range(1, 21)]  # 20 unique clients
    
    data = []
    
    today = datetime.now()
    
    for i in range(num_campaigns):
        campaign_id = f'CMP-{1000+i}'
        client = random.choice(clients)
        platform = random.choice(platforms)
        
        # Random start date within the last 30 days to next 30 days
        start_date = today - timedelta(days=random.randint(1, 60))
        duration = random.randint(14, 90) # Campaign duration between 2 weeks and 3 months
        end_date = start_date + timedelta(days=duration)
        
        # Budget and Goals
        budget = round(np.random.uniform(5000, 50000), 2)
        cpm = np.random.uniform(10, 25) # Cost Per Mille (thousand impressions)
        impressions_goal = int((budget / cpm) * 1000)
        
        # Simulate delivery based on elapsed time (with some variance)
        total_days = (end_date - start_date).days
        days_elapsed = (today - start_date).days
        days_elapsed = max(0, min(days_elapsed, total_days)) # Clamp between 0 and total duration
        
        if days_elapsed == 0:
            impressions_delivered = 0
        else:
            # Expected delivery if linear
            expected_delivery_pct = days_elapsed / total_days
            
            # Introduce variance to create "At-Risk" campaigns (0.7 to 1.3 pacing factor)
            variance = np.random.normal(1.0, 0.2) 
            actual_delivery_pct = expected_delivery_pct * variance
            impressions_delivered = int(impressions_goal * actual_delivery_pct)
            
        data.append({
            'Campaign_ID': campaign_id,
            'Client_Name': client,
            'Platform': platform,
            'Impressions_Goal': impressions_goal,
            'Impressions_Delivered': impressions_delivered,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Budget': budget,
            'Total_Days': total_days,
            'Days_Elapsed': days_elapsed
        })
        
    return pd.DataFrame(data)

# Generate the dataframe
df = generate_synthetic_data()
print("Dataset Generated Successfully!")
print(df.head())

# --- PART 2: Pacing Logic & Anomaly Detection ---

def calculate_pacing(row):
    if row['Days_Elapsed'] == 0:
        return 0.0
    
    goal_pacing_pct = row['Days_Elapsed'] / row['Total_Days']
    actual_pacing_pct = row['Impressions_Delivered'] / row['Impressions_Goal']
    
    if goal_pacing_pct == 0:
        return 0.0
        
    # Pacing formula: (Actual % / Expected %)
    # 1.0 = On Track (100%), < 1.0 = Underpacing, > 1.0 = Overpacing
    pacing_index = actual_pacing_pct / goal_pacing_pct
    return pacing_index

df['Pacing_Index'] = df.apply(calculate_pacing, axis=1)
df['Pacing_%'] = (df['Pacing_Index'] * 100).round(2)

# Define 'At-Risk' criteria: Pacing < 90% or > 110%
# Also filtering out campaigns that haven't started or are just starting (Days Elapsed > 5 to be significant)
df['Status'] = 'On Track'
df.loc[(df['Pacing_%'] < 90) & (df['Days_Elapsed'] > 5), 'Status'] = 'Under-Pacing (At Risk)'
df.loc[(df['Pacing_%'] > 110) & (df['Days_Elapsed'] > 5), 'Status'] = 'Over-Pacing (At Risk)'

at_risk_campaigns = df[df['Status'].str.contains('At Risk')].copy()

print(f"\nIdentified {len(at_risk_campaigns)} At-Risk Campaigns.")

# --- PART 3: AI Insight Agent (Simulated) ---
# In a real production environment, this would call an API like OpenAI's GPT-4 or Google Gemini.
# For this Portfolio Project, we simulate the 'Agent' behavior with rule-based text generation to demonstrate the concept.

def ai_insight_agent(row):
    status = row['Status']
    pacing = row['Pacing_%']
    platform = row['Platform']
    budget = row['Budget']
    
    if "Under-Pacing" in status:
        severity = "Critical" if pacing < 70 else "Moderate"
        return (f"⚠️ **AI ALERT ({severity})**: Campaign is under-pacing at {pacing}%. "
                f"Recommendation: Increase daily bid caps on {platform} inventory immediately. "
                "Consider reallocating budget to high-performing dayparts to ensure full delivery before end date.")
    elif "Over-Pacing" in status:
        return (f"⚠️ **AI ALERT**: Campaign is burning budget too fast ({pacing}% pacing). "
                f"Recommendation: Tighten frequency caps on {platform} or pause lower-performing creatives. "
                "Check for bot traffic anomalies if CTR is unexpectedly high.")
    else:
        return "Campaign is on track."

# Apply the AI Agent to the At-Risk dataframe
at_risk_campaigns['AI_Recommendation'] = at_risk_campaigns.apply(ai_insight_agent, axis=1)

# Display a sample of the AI insights
print("\n--- AI Agent Insights (Sample) ---")
for index, row in at_risk_campaigns.head(3).iterrows():
    print(f"Campaign: {row['Campaign_ID']} | Status: {row['Status']}")
    print(f"Recommendation: {row['AI_Recommendation']}\n")


# --- PART 4: Visualization (Revenue at Risk) ---

# Calculate 'Revenue at Risk' (Simplified as total budget of at-risk campaigns)
# In reality, this might be the unspent portion, but for the chart we visualize Total Budget at Risk by Platform.

# Create a summary table for the chart
risk_summary = at_risk_campaigns.groupby(['Platform', 'Status'])['Budget'].sum().reset_index()
risk_summary.rename(columns={'Budget': 'Revenue_at_Risk'}, inplace=True)

# Create Bar Chart using Plotly
fig = px.bar(
    risk_summary, 
    x='Platform', 
    y='Revenue_at_Risk', 
    color='Status',
    title='💰 Revenue at Risk by Platform (Under/Over Pacing)',
    text_auto='.2s',
    color_discrete_map={
        'Under-Pacing (At Risk)': '#FF5733',  # Red/Orange for under
        'Over-Pacing (At Risk)': '#FFC300'    # Yellow for over
    },
    template='plotly_white'
)

fig.update_layout(yaxis_title="Total Budget (USD)")

# Show the interactive chart (in Colab this works automatically)
fig.show()

# Show the Summary Data Table
print("\n--- At-Risk Campaigns Data Summary ---")
print(at_risk_campaigns[['Campaign_ID', 'Platform', 'Budget', 'Pacing_%', 'Status']].head(10))
