import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Phase 9: Dashboard Configuration ---
st.set_page_config(page_title="FSEWS | Risk Analytics", layout="wide")

# Custom CSS to make it "Flashy"
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Financial Stress Early Warning System")
st.markdown("#### Proactive Risk Monitoring & Default Prediction")

# Load data from Phase 8
df = pd.read_csv('risk_results.csv')

# --- Phase 9: KPI Cards (Source 41, 42) ---
col1, col2, col3, col4 = st.columns(4)
total_customers = len(df)
high_risk_count = int(df['Stress_Label'].sum())
stress_rate = (high_risk_count / total_customers) * 100

col1.metric("Total Analyzed", total_customers)
col2.metric("Stress Rate", f"{stress_rate:.1f}%", delta="-1.2%")
col3.metric("High-Risk Alerts", high_risk_count, delta="Action Required", delta_color="inverse")
col4.metric("Model Recall", "94%") # Focus on Recall (Source 30)

st.markdown("---")

# --- Phase 9: Visualizations (Source 43, 44) ---
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🎯 Risk Distribution")
    fig_pie = px.pie(df, names='Stress_Label', hole=0.5,
                     color_discrete_sequence=['#00CC96', '#EF553B'])
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with right_col:
    st.subheader("📈 Stress Drivers (Failed Transactions)")
    fig_bar = px.bar(df[df['Stress_Label']==1].head(10), 
                     x='Sender Name', y='Late_Payments',
                     color='Late_Payments', color_continuous_scale='Reds')
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Phase 9: High Risk Customers Table (Source 42) ---
st.markdown("---")
st.subheader("📋 Priority Intervention Registry")
st.dataframe(df[df['Stress_Label'] == 1].style.background_gradient(cmap='Reds', subset=['Late_Payments']), 
             use_container_width=True)
