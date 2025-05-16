"""
Model Metrics Dashboard page for the SilentCodingLegend AI application.
Displays metrics on model usage, performance, and costs.
"""

import streamlit as st

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Model Metrics | SilentCodingLegend AI",
    page_icon="📊",
    layout="wide",
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from io import StringIO, BytesIO
import json
from src.model_metrics import metrics_tracker
from src.theme import apply_theme_style

# Apply custom theme
apply_theme_style()

# Header
st.title("📊 Model Performance Metrics")
st.markdown("""
This dashboard shows metrics related to your model usage, including:
- Response times
- Token usage
- Cost estimates
- Cache effectiveness
- Success rates
- Model comparisons
""")

# Date filtering
st.sidebar.header("Filter Options")
today = datetime.datetime.now()
one_week_ago = today - datetime.timedelta(days=7)
one_month_ago = today - datetime.timedelta(days=30)

filter_options = ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"]
selected_filter = st.sidebar.selectbox("Time Period", filter_options)

date_from = None
date_to = None

if selected_filter == "Last 7 Days":
    date_from = one_week_ago
elif selected_filter == "Last 30 Days":
    date_from = one_month_ago
elif selected_filter == "Last 90 Days":
    date_from = today - datetime.timedelta(days=90)
elif selected_filter == "Custom Range":
    col1, col2 = st.sidebar.columns(2)
    date_from = col1.date_input("From", one_week_ago.date())
    date_to = col2.date_input("To", today.date())
    date_from = datetime.datetime.combine(date_from, datetime.time.min)
    date_to = datetime.datetime.combine(date_to, datetime.time.max)

# Model filtering
models = ["All Models", "llama-3.1-70b", "llama-3.1-8b", "llama-3.1-70b-vision"]
selected_model = st.sidebar.selectbox("Model Filter", models)
model_filter = None if selected_model == "All Models" else selected_model

# Get metrics data
metrics = metrics_tracker.get_metrics(
    limit=1000,
    model_filter=model_filter,
    date_from=date_from,
    date_to=date_to
)

# Display metrics visualizations
if not metrics:
    st.info("No metrics data available for the selected filters. Make some queries first!")
else:
    # Create a pandas DataFrame for the metrics
    df = pd.DataFrame(metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    
    # Display metrics summary
    summary = metrics_tracker.get_metrics_summary()
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Comparison", "Cost Analysis", "Detailed Data"])
    
    with tab1:
        # Overview Tab
        st.header("Overview Dashboard")
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", summary["total_queries"])
        with col2:
            st.metric("Avg Response Time", f"{summary['avg_response_time']:.2f}s")
        with col3:
            st.metric("Total Tokens", f"{summary['total_tokens']:,}")
        with col4:
            st.metric("Total Cost", f"${summary['total_cost']:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
        with col2:
            st.metric("Cache Hit Rate", f"{summary['cache_hit_rate']:.1f}%")
        
        # Time series charts
        st.subheader("Activity Over Time")
        
        # Get time-based metrics
        time_period = "day"  # default to day
        time_metrics = metrics_tracker.get_metrics_by_time_period(
            period_type=time_period, 
            model_filter=model_filter
        )
        
        if not time_metrics.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Query volume over time
                fig = px.bar(
                    time_metrics, 
                    x="period", 
                    y="num_queries",
                    title="Number of Queries by Day"
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Queries")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Response time over time
                fig = px.line(
                    time_metrics, 
                    x="period", 
                    y="response_time",
                    title="Average Response Time by Day"
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Response Time (s)")
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Token usage over time
                fig = px.area(
                    time_metrics, 
                    x="period", 
                    y="total_tokens",
                    title="Token Usage by Day"
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Total Tokens")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cost over time
                fig = px.line(
                    time_metrics, 
                    x="period", 
                    y="estimated_cost",
                    title="Estimated Cost by Day"
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Cost ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Model distribution
        st.subheader("Model Usage Distribution")
        model_df = pd.DataFrame([
            {
                "model": model,
                "queries": stats["count"],
                "tokens": stats["total_tokens"],
                "cost": stats["total_cost"]
            }
            for model, stats in summary.get("model_stats", {}).items()
        ])
        
        if not model_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    model_df, 
                    values="queries", 
                    names="model", 
                    title="Queries by Model",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    model_df, 
                    values="tokens", 
                    names="model", 
                    title="Token Usage by Model",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Model Comparison Tab
        st.header("Model Comparison")
        st.markdown("""
        Compare performance metrics between different models to determine which offers the best 
        performance for your use case. This can help you make informed decisions about model selection.
        """)
        
        model_comparison = metrics_tracker.get_model_comparison()
        
        if model_comparison.empty:
            st.info("Not enough data to compare models. Try using different models first.")
        else:
            # Format the dataframe for display
            display_df = model_comparison.copy()
            
            # Format columns for readability
            display_df["response_time_mean"] = display_df["response_time_mean"].round(2).astype(str) + " s"
            display_df["estimated_cost_mean"] = "$" + display_df["estimated_cost_mean"].round(6).astype(str)
            display_df["estimated_cost_sum"] = "$" + display_df["estimated_cost_sum"].round(4).astype(str)
            display_df["success_mean"] = (display_df["success_mean"] * 100).round(1).astype(str) + "%"
            display_df["is_cached_mean"] = (display_df["is_cached_mean"] * 100).round(1).astype(str) + "%"
            display_df["tokens_per_dollar"] = display_df["tokens_per_dollar"].round(0).astype(int)
            display_df["tokens_per_second"] = display_df["tokens_per_second"].round(1)
            
            # Select and rename columns for display
            display_cols = {
                "model_id": "Model",
                "response_time_mean": "Avg Response Time",
                "total_tokens_mean": "Avg Tokens",
                "estimated_cost_mean": "Avg Cost",
                "estimated_cost_sum": "Total Cost", 
                "success_mean": "Success Rate",
                "is_cached_mean": "Cache Hit Rate",
                "num_queries": "# Queries",
                "tokens_per_dollar": "Tokens per $",
                "tokens_per_second": "Tokens per Second"
            }
            display_df = display_df.rename(columns=display_cols)[display_cols.values()]
            
            # Display the comparison table
            st.dataframe(display_df, use_container_width=True)
            
            # Display comparative charts
            st.subheader("Performance Comparison Charts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time by model
                fig = px.bar(
                    model_comparison, 
                    x="model_id", 
                    y="response_time_mean",
                    title="Average Response Time by Model"
                )
                fig.update_layout(xaxis_title="Model", yaxis_title="Response Time (s)")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Token efficiency (tokens per $)
                fig = px.bar(
                    model_comparison, 
                    x="model_id", 
                    y="tokens_per_dollar",
                    title="Cost Efficiency (Tokens per $)"
                )
                fig.update_layout(xaxis_title="Model", yaxis_title="Tokens per Dollar")
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Tokens per second
                fig = px.bar(
                    model_comparison, 
                    x="model_id", 
                    y="tokens_per_second",
                    title="Speed Efficiency (Tokens per Second)"
                )
                fig.update_layout(xaxis_title="Model", yaxis_title="Tokens per Second")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Success rate
                fig = px.bar(
                    model_comparison, 
                    x="model_id", 
                    y="success_mean",
                    title="Success Rate by Model"
                )
                fig.update_layout(
                    xaxis_title="Model", 
                    yaxis_title="Success Rate",
                    yaxis=dict(tickformat='.0%')
                )
                st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        # Cost Analysis Tab
        st.header("Cost Analysis")
        st.markdown("""
        Analyze your model usage costs over time and by model. This can help you 
        optimize your usage and budget.
        """)
        
        # Get cost breakdown data
        days_to_analyze = 30
        if selected_filter == "Last 7 Days":
            days_to_analyze = 7
        elif selected_filter == "Last 90 Days":
            days_to_analyze = 90
            
        cost_data = metrics_tracker.get_cost_breakdown(days=days_to_analyze)
        
        # Display total cost prominently
        st.metric(
            "Total Estimated Cost", 
            f"${cost_data['total_cost']:.4f}",
            help="Total estimated cost based on token usage and model pricing"
        )
        
        # Cost over time
        st.subheader("Cost Over Time")
        
        if cost_data['daily_costs']:
            daily_costs_df = pd.DataFrame(cost_data['daily_costs'])
            daily_costs_df['date'] = pd.to_datetime(daily_costs_df['date'])
            daily_costs_df = daily_costs_df.sort_values('date')
            
            fig = px.line(
                daily_costs_df, 
                x="date", 
                y="estimated_cost",
                title="Daily Cost",
                markers=True
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Cost ($)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cost data available for the selected time period.")
        
        # Cost by model
        st.subheader("Cost by Model")
        
        if cost_data['model_costs']:
            model_costs_df = pd.DataFrame([
                {"model": model, "cost": cost}
                for model, cost in cost_data['model_costs'].items()
            ])
            
            fig = px.pie(
                model_costs_df,
                values="cost",
                names="model",
                title="Cost Distribution by Model",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost by query type
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost by Query Type")
            
            if cost_data['query_type_costs']:
                query_costs_df = pd.DataFrame([
                    {"query_type": query_type, "cost": cost}
                    for query_type, cost in cost_data['query_type_costs'].items()
                ])
                
                fig = px.pie(
                    query_costs_df,
                    values="cost",
                    names="query_type",
                    title="Cost by Query Type",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cost Projection")
            
            if cost_data['daily_costs']:
                daily_costs_df = pd.DataFrame(cost_data['daily_costs'])
                if not daily_costs_df.empty:
                    avg_daily_cost = daily_costs_df['estimated_cost'].mean()
                    
                    projection_days = st.slider("Project costs for next X days:", 
                                             min_value=7, max_value=365, value=30)
                    
                    projected_cost = avg_daily_cost * projection_days
                    
                    st.metric(
                        f"Projected Cost (Next {projection_days} Days)",
                        f"${projected_cost:.4f}"
                    )
                    
                    # Create a simple projection chart
                    last_date = daily_costs_df['date'].max()
                    if isinstance(last_date, str):
                        last_date = pd.to_datetime(last_date)
                        
                    future_dates = pd.date_range(
                        start=last_date, 
                        periods=projection_days+1, 
                        freq='D'
                    )[1:]
                    
                    projection_df = pd.DataFrame({
                        'date': future_dates,
                        'projected_cost': [avg_daily_cost] * len(future_dates)
                    })
                    
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=daily_costs_df['date'],
                        y=daily_costs_df['estimated_cost'],
                        mode='lines+markers',
                        name='Historical Cost',
                        line=dict(color='#7e57c2')
                    ))
                    
                    # Add projection
                    fig.add_trace(go.Scatter(
                        x=projection_df['date'],
                        y=projection_df['projected_cost'],
                        mode='lines',
                        name='Projected Cost',
                        line=dict(color='#4fc3f7', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Cost Projection",
                        xaxis_title="Date",
                        yaxis_title="Cost ($)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Raw Data Tab
        st.header("Detailed Metrics Data")
        
        # Recent API calls table with pagination
        st.subheader("Recent API Calls")
        
        page_size = st.slider("Rows per page:", 10, 100, 25)
        
        # Simple pagination
        total_pages = (len(df) + page_size - 1) // page_size
        if total_pages > 0:
            page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            
            page_df = df.iloc[start_idx:end_idx].copy()
            
            # Format for display
            display_df = page_df[["timestamp", "model_id", "response_time", "total_tokens", 
                           "estimated_cost", "is_cached", "success", "query_type"]].copy()
            
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_df["estimated_cost"] = display_df["estimated_cost"].apply(lambda x: f"${x:.6f}")
            display_df["response_time"] = display_df["response_time"].apply(lambda x: f"{x:.3f}s")
            
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown(f"Showing {start_idx+1}-{end_idx} of {len(df)} entries")
    
    # Advanced options
    with st.expander("Advanced Options"):
        st.subheader("Export Metrics")
        
        export_tabs = st.tabs(["CSV", "JSON", "Excel", "HTML"])
        
        with export_tabs[0]:
            if st.button("Generate CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="model_metrics.csv",
                    mime="text/csv"
                )
        
        with export_tabs[1]:
            if st.button("Generate JSON"):
                json_data = json.dumps(metrics, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="model_metrics.json",
                    mime="application/json"
                )
        
        with export_tabs[2]:
            if st.button("Generate Excel"):
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    df.to_excel(writer, sheet_name="Metrics Data", index=False)
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame([{
                        "Total Queries": summary["total_queries"],
                        "Avg Response Time": summary["avg_response_time"],
                        "Total Tokens": summary["total_tokens"],
                        "Total Cost": summary["total_cost"],
                        "Success Rate": summary["success_rate"],
                        "Cache Hit Rate": summary["cache_hit_rate"]
                    }])
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    
                    # Add model comparison sheet
                    model_comparison = metrics_tracker.get_model_comparison()
                    if not model_comparison.empty:
                        model_comparison.to_excel(writer, sheet_name="Model Comparison", index=False)
                
                excel_data = buffer.getvalue()
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name="model_metrics.xlsx",
                    mime="application/vnd.ms-excel"
                )
        
        with export_tabs[3]:
            if st.button("Generate HTML Report"):
                # Create a simple HTML report
                html_io = StringIO()
                html_io.write("<html><head><title>Model Metrics Report</title>")
                html_io.write("<style>body{font-family:Arial,sans-serif;margin:20px}table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}th{background-color:#f2f2f2}</style>")
                html_io.write("</head><body>")
                
                # Add title and date
                html_io.write(f"<h1>Model Metrics Report</h1>")
                html_io.write(f"<p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                
                # Add summary
                html_io.write("<h2>Summary</h2>")
                html_io.write("<table>")
                html_io.write("<tr><th>Metric</th><th>Value</th></tr>")
                html_io.write(f"<tr><td>Total Queries</td><td>{summary['total_queries']}</td></tr>")
                html_io.write(f"<tr><td>Avg Response Time</td><td>{summary['avg_response_time']:.2f}s</td></tr>")
                html_io.write(f"<tr><td>Total Tokens</td><td>{summary['total_tokens']:,}</td></tr>")
                html_io.write(f"<tr><td>Total Cost</td><td>${summary['total_cost']:.4f}</td></tr>")
                html_io.write(f"<tr><td>Success Rate</td><td>{summary['success_rate']:.1f}%</td></tr>")
                html_io.write(f"<tr><td>Cache Hit Rate</td><td>{summary['cache_hit_rate']:.1f}%</td></tr>")
                html_io.write("</table>")
                
                # Add data table
                html_io.write("<h2>Recent API Calls</h2>")
                html_io.write(df.head(100).to_html(index=False))
                
                html_io.write("</body></html>")
                
                st.download_button(
                    label="Download HTML Report",
                    data=html_io.getvalue(),
                    file_name="model_metrics_report.html",
                    mime="text/html"
                )
        
        st.subheader("Reset Metrics")
        if st.button("Reset All Metrics", type="primary"):
            # This is a destructive operation, so we add a confirmation
            st.warning("This will delete all collected metrics data. Are you sure?")
            confirm = st.button("Yes, I'm sure")
            if confirm:
                import os
                if os.path.exists(metrics_tracker.storage_path):
                    os.remove(metrics_tracker.storage_path)
                    metrics_tracker._ensure_storage_exists()
                    st.success("Metrics data has been reset.")
                    st.rerun()

# Footer with refresh button
st.markdown("---")
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("Refresh"):
        st.rerun()
