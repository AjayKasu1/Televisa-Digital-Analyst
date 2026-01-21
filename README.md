# TelevisaUnivision Digital Analyst Portfolio Project

##  Project Overview
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://televisa-digital-analyst-nezyypeprqvzbkksenvgwq.streamlit.app/)

**🔴 [View Live Demo Here](https://televisa-digital-analyst-nezyypeprqvzbkksenvgwq.streamlit.app/)**

This project demonstrates automated reporting, data engineering, and visualization capabilities for a Digital Analyst role. It simulates a digital ad campaign environment to generate insights, calculate pacing metrics, and identify at-risk campaigns using Python.

## Key Features
- **Automated Data Generation**: Creates a synthetic dataset of 100 digital ad campaigns with realistic parameters (Impressions, Budget, Platforms).
- **Pacing & Anomaly Detection**: Implements business logic to calculate 'Pacing %' and automatically flag campaigns that are under-pacing (<90%) or over-pacing (>110%).
- **AI Insight Agent**: Simulates an AI-driven recommendation engine that prescribes specific actions (e.g., "Increase bid caps") based on campaign status.
- **Interactive Dashboards**: detailed Plotly visualizations including:
    -   Revenue at Risk Analysis
    -   Deadline Urgency Heatmap
    -   "High Stakes" Client Scatter Plot
    -   eCPM Efficiency Gauge

- `televisa_colab_project.py`: The main Python script. Designed to be run locally or in Google Colab. contains all logic for data generation, analysis, and visualization.
- `app.py`: The Streamlit web application.
- `simulated_data/`: Folder containing the exported CSV dataset (`televisa_campaign_data.csv`).

## How to Run
### Option 1: Streamlit App (Recommended)
This launches the full interactive web application.
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Local Python Script
```bash
pip install pandas plotly numpy
python televisa_colab_project.py
```
2.  **Google Colab**:
    Simply copy the contents of `televisa_colab_project.py` into a Google Colab cell and execute.

## Visualizations
The project generates an interactive dashboard highlighting:
- **Revenue at Risk**: Total budget tied to at-risk campaigns, broken down by platform.
- **Urgency Heatmap**: Identifying which platforms are falling behind as deadlines approach.
- **Client Health**: A scatter plot identifying large-budget clients that are off-track.

---
*Created by [Your Name] for TelevisaUnivision Digital Analyst Application.*
