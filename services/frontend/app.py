import streamlit as st
import pandas as pd
import altair as alt
import os
import time

st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="🧠",
    layout="wide",
)

# Configuration
# Path to logs relative to this file
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated", "logs")
GLOBAL_METRICS_FILE = os.path.join(LOGS_DIR, "global_metrics.csv")
CLIENT_METRICS_FILE = os.path.join(LOGS_DIR, "client_metrics.csv")

st.title("🧠 Federated Learning Dashboard")

def load_data():
    global_df = pd.DataFrame()
    client_df = pd.DataFrame()
    
    if os.path.exists(GLOBAL_METRICS_FILE):
        try:
            global_df = pd.read_csv(GLOBAL_METRICS_FILE)
        except Exception as e:
            st.error(f"Error reading global metrics: {e}")
            
    if os.path.exists(CLIENT_METRICS_FILE):
        try:
            client_df = pd.read_csv(CLIENT_METRICS_FILE)
        except Exception as e:
            st.error(f"Error reading client metrics: {e}")
            
    return global_df, client_df

# Sidebar for auto-refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
if auto_refresh:
    time.sleep(2)
    st.rerun()

global_df, client_df = load_data()

if global_df.empty and client_df.empty:
    st.warning("No metrics found yet. Waiting for training to start...")
else:
    # --- Global Metrics Section ---
    st.header("Global Aggregated Metrics")
    
    if not global_df.empty:
        # Create tabs for Fit vs Evaluate
        tab1, tab2 = st.tabs(["Evaluation (Global Test)", "Fit (Training Aggregation)"])
        
        with tab1:
            st.subheader("Evaluation Metrics")
            eval_df = global_df[global_df["stage"] == "evaluate"]
            if not eval_df.empty:
                # Get unique metrics
                metrics = eval_df["metric_name"].unique()
                
                # Check for "Best" metrics vs regular
                # We can group by metric name
                
                # Let's plot main metrics
                selected_metrics = st.multiselect("Select Metrics", metrics, default=[m for m in metrics if "best" not in m and "min" not in m and "max" not in m])
                
                if selected_metrics:
                    chart_data = eval_df[eval_df["metric_name"].isin(selected_metrics)]
                    
                    chart = alt.Chart(chart_data).mark_line(point=True).encode(
                        x='round:O',
                        y='value:Q',
                        color='metric_name:N',
                        tooltip=['round', 'metric_name', 'value']
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                
                # Show Min/Max/Avg for specific metric
                st.markdown("### Detailed Metric Analysis")
                target_metric = st.selectbox("Select Metric for Detail", [m for m in metrics if "_avg" in m or (m not in [x + "_min" for x in metrics] and m not in [x + "_max" for x in metrics])])
                
                # Try to find base name
                base_name = target_metric.replace("_avg", "")
                suffixes = ["_min", "_avg", "_max", "_best"]
                related_metrics = [base_name + s for s in suffixes if base_name + s in metrics]
                if target_metric in metrics and target_metric not in related_metrics: 
                     # Handle case where metric doesn't have suffix
                     related_metrics.append(target_metric)
                
                if related_metrics:
                    detail_data = eval_df[eval_df["metric_name"].isin(related_metrics)]
                    detail_chart = alt.Chart(detail_data).mark_line().encode(
                        x='round:O',
                        y='value:Q',
                        color='metric_name:N',
                        strokeDash=alt.condition(
                            alt.datum.metric_name.endswith('_best'),
                            alt.value([5, 5]),  # dashed line for 'best'
                            alt.value([0])      # solid line for others
                        ),
                        tooltip=['round', 'metric_name', 'value']
                    ).interactive()
                    st.altair_chart(detail_chart, use_container_width=True)

            else:
                st.info("No evaluation metrics yet.")

        with tab2:
            st.subheader("Fit Metrics (Aggregated Training)")
            fit_df = global_df[global_df["stage"] == "fit"]
            if not fit_df.empty:
                metrics = fit_df["metric_name"].unique()
                selected_metrics_fit = st.multiselect("Select Metrics (Fit)", metrics, default=[m for m in metrics if "loss" in m])
                
                if selected_metrics_fit:
                    chart_data = fit_df[fit_df["metric_name"].isin(selected_metrics_fit)]
                    chart = alt.Chart(chart_data).mark_line(point=True).encode(
                        x='round:O',
                        y='value:Q',
                        color='metric_name:N',
                        tooltip=['round', 'metric_name', 'value']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No fit metrics yet.")

    # --- Client Metrics Section ---
    st.header("Client Individual Metrics")
    
    if not client_df.empty:
        stage_filter = st.radio("Select Stage", ["evaluate", "fit"], index=0, horizontal=True)
        filtered_client_df = client_df[client_df["stage"] == stage_filter]
        
        if not filtered_client_df.empty:
            client_metrics = filtered_client_df["metric_name"].unique()
            selected_client_metric = st.selectbox("Select Client Metric", client_metrics)
            
            client_chart_data = filtered_client_df[filtered_client_df["metric_name"] == selected_client_metric]
            
            # Client Performance Over Time
            st.subheader(f"Client Performance: {selected_client_metric}")
            
            c_chart = alt.Chart(client_chart_data).mark_line(point=True).encode(
                x='round:O',
                y='value:Q',
                color='client_id:N',
                tooltip=['round', 'client_id', 'value']
            ).interactive()
            st.altair_chart(c_chart, use_container_width=True)
            
            # Data Table
            st.dataframe(client_chart_data.pivot(index=["round", "client_id"], columns="metric_name", values="value"))

        else:
            st.info(f"No client metrics for {stage_filter} stage.")
