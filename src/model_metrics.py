"""
Model Performance Metrics for SilentCodingLegend AI.
This module tracks and analyzes performance metrics for LLM API calls.
"""

import time
import json
import os
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

# Define constants for cost calculation
MODEL_COST_PER_TOKEN = {
    "llama-3.1-70b": {
        "input": 0.00000015,  # $0.15 / 1M tokens
        "output": 0.00000045,  # $0.45 / 1M tokens
    },
    "llama-3.1-8b": {
        "input": 0.00000008,  # $0.08 / 1M tokens
        "output": 0.00000024,  # $0.24 / 1M tokens
    },
    "llama-3.1-70b-vision": {
        "input": 0.00000025,  # $0.25 / 1M tokens
        "output": 0.00000075,  # $0.75 / 1M tokens
    },
    # Add other models as needed
}

# Default cost for unknown models
DEFAULT_COST_PER_TOKEN = {
    "input": 0.00000020,
    "output": 0.00000060,
}

class ModelMetricsTracker:
    """Track and analyze model performance metrics."""
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the metrics tracker.
        
        Args:
            storage_path: Path to store metrics data. Defaults to 'data/metrics.json'
        """
        if storage_path is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(base_dir, exist_ok=True)
            storage_path = os.path.join(base_dir, 'metrics.json')
        
        self.storage_path = storage_path
        self._ensure_storage_exists()
        
    def _ensure_storage_exists(self) -> None:
        """Ensure the metrics storage file exists."""
        if not os.path.exists(self.storage_path):
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            # Initialize empty metrics file
            with open(self.storage_path, 'w') as f:
                json.dump([], f)
    
    def start_tracking(self) -> float:
        """Start tracking a new API call.
        
        Returns:
            float: Start timestamp
        """
        return time.time()
    
    def record_metrics(self, 
                       model_id: str, 
                       start_time: float,
                       input_tokens: int,
                       output_tokens: int,
                       total_tokens: Optional[int] = None,
                       is_cached: bool = False,
                       query_type: str = "text",
                       success: bool = True,
                       error_type: Optional[str] = None,
                       template_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Record metrics for a completed API call.
        
        Args:
            model_id: The ID of the model used
            start_time: Start timestamp from start_tracking()
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens (input + output) if available
            is_cached: Whether the response was from cache
            query_type: Type of query (text, vision, etc.)
            success: Whether the API call was successful
            error_type: Type of error if not successful
            template_id: ID of the chat template used, if any
            
        Returns:
            Dict containing the recorded metrics
        """
        end_time = time.time()
        response_time = end_time - start_time
        
        # Calculate total tokens if not provided
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost = self._calculate_cost(model_id, input_tokens, output_tokens)
        
        # Build metrics object
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": model_id,
            "response_time": response_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": cost,
            "is_cached": is_cached,
            "query_type": query_type,
            "success": success,
            "error_type": error_type,
            "template_id": template_id
        }
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the estimated cost of an API call.
        
        Args:
            model_id: The ID of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        # Get cost rates for the model
        cost_rates = MODEL_COST_PER_TOKEN.get(model_id, DEFAULT_COST_PER_TOKEN)
        
        # Calculate cost
        input_cost = input_tokens * cost_rates["input"]
        output_cost = output_tokens * cost_rates["output"]
        total_cost = input_cost + output_cost
        
        return total_cost
    
    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save metrics to storage.
        
        Args:
            metrics: Metrics to save
        """
        try:
            # Load existing metrics
            with open(self.storage_path, 'r') as f:
                existing_metrics = json.load(f)
            
            # Append new metrics
            existing_metrics.append(metrics)
            
            # Save back to file
            with open(self.storage_path, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
        except Exception as e:
            st.warning(f"Failed to save metrics: {str(e)}")
    
    def get_metrics(self, 
                    limit: int = 100, 
                    model_filter: Optional[str] = None,
                    date_from: Optional[datetime.datetime] = None,
                    date_to: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """
        Get stored metrics with optional filtering.
        
        Args:
            limit: Maximum number of records to return
            model_filter: Filter by model ID
            date_from: Filter by start date
            date_to: Filter by end date
            
        Returns:
            List of metric records
        """
        try:
            with open(self.storage_path, 'r') as f:
                all_metrics = json.load(f)
            
            # Apply filters
            filtered_metrics = all_metrics
            
            if model_filter:
                filtered_metrics = [m for m in filtered_metrics if m["model_id"] == model_filter]
            
            if date_from:
                date_from_str = date_from.isoformat()
                filtered_metrics = [m for m in filtered_metrics if m["timestamp"] >= date_from_str]
            
            if date_to:
                date_to_str = date_to.isoformat()
                filtered_metrics = [m for m in filtered_metrics if m["timestamp"] <= date_to_str]
            
            # Sort by timestamp (newest first) and limit
            sorted_metrics = sorted(filtered_metrics, key=lambda x: x["timestamp"], reverse=True)
            limited_metrics = sorted_metrics[:limit]
            
            return limited_metrics
        except Exception as e:
            st.warning(f"Failed to retrieve metrics: {str(e)}")
            return []
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dict containing summary statistics
        """
        metrics = self.get_metrics(limit=1000)  # Get a large sample for the summary
        
        if not metrics:
            return {
                "total_queries": 0,
                "avg_response_time": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "success_rate": 0,
                "cache_hit_rate": 0
            }
        
        # Calculate summary statistics
        total_queries = len(metrics)
        avg_response_time = sum(m["response_time"] for m in metrics) / total_queries if total_queries else 0
        total_tokens = sum(m["total_tokens"] for m in metrics)
        total_cost = sum(m["estimated_cost"] for m in metrics)
        success_count = sum(1 for m in metrics if m["success"])
        cache_hit_count = sum(1 for m in metrics if m["is_cached"])
        
        success_rate = (success_count / total_queries) * 100 if total_queries else 0
        cache_hit_rate = (cache_hit_count / total_queries) * 100 if total_queries else 0
        
        # Group by model
        model_stats = {}
        for m in metrics:
            model_id = m["model_id"]
            if model_id not in model_stats:
                model_stats[model_id] = {
                    "count": 0,
                    "total_tokens": 0,
                    "total_cost": 0
                }
            model_stats[model_id]["count"] += 1
            model_stats[model_id]["total_tokens"] += m["total_tokens"]
            model_stats[model_id]["total_cost"] += m["estimated_cost"]
        
        return {
            "total_queries": total_queries,
            "avg_response_time": avg_response_time,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "model_stats": model_stats
        }
    
    def visualize_metrics(self) -> None:
        """
        Create and display visualizations of metrics in Streamlit.
        """
        st.header("Model Performance Metrics")
        
        # Get metrics and summary
        metrics = self.get_metrics(limit=500)
        summary = self.get_metrics_summary()
        
        if not metrics:
            st.info("No metrics data available yet. Make some API calls to collect data.")
            return
        
        # Create a pandas DataFrame for easier visualization
        df = pd.DataFrame(metrics)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        
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
        
        # Model comparison
        st.subheader("Model Comparison")
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
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Usage by Model", "Response Time", "Token Usage"])
            
            with tab1:
                fig = px.pie(model_df, values="queries", names="model", title="Queries by Model")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Group by date and model for timeseries
                time_df = df.groupby(["date", "model_id"])["response_time"].mean().reset_index()
                fig = px.line(time_df, x="date", y="response_time", color="model_id", 
                             title="Average Response Time by Date")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Token usage over time
                token_df = df.groupby(["date"])["total_tokens"].sum().reset_index()
                fig = px.bar(token_df, x="date", y="total_tokens",
                           title="Total Token Usage by Date")
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent queries table
        st.subheader("Recent API Calls")
        recent_df = df[["timestamp", "model_id", "response_time", "total_tokens", 
                       "estimated_cost", "is_cached", "success"]].head(10)
        recent_df["timestamp"] = recent_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        recent_df["estimated_cost"] = recent_df["estimated_cost"].apply(lambda x: f"${x:.6f}")
        recent_df["response_time"] = recent_df["response_time"].apply(lambda x: f"{x:.3f}s")
        st.dataframe(recent_df, use_container_width=True)
    
    def get_metrics_by_time_period(self, 
                                  period_type: str = "day",
                                  limit_days: int = 30,
                                  model_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Get metrics aggregated by time period (day, week, month).
        
        Args:
            period_type: Type of period to aggregate by ('day', 'week', 'month')
            limit_days: Limit to recent number of days
            model_filter: Optional filter by model ID
            
        Returns:
            DataFrame with metrics aggregated by time period
        """
        metrics = self.get_metrics(limit=10000)  # Get a large sample
        
        if not metrics:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by date range
        if limit_days:
            start_date = pd.Timestamp.now() - pd.Timedelta(days=limit_days)
            df = df[df["timestamp"] >= start_date]
            
        # Filter by model if specified
        if model_filter:
            df = df[df["model_id"] == model_filter]
            
        # Create period column based on period_type
        if period_type == "day":
            df["period"] = df["timestamp"].dt.date
        elif period_type == "week":
            df["period"] = df["timestamp"].dt.to_period("W").dt.start_time
        elif period_type == "month":
            df["period"] = df["timestamp"].dt.to_period("M").dt.start_time
        else:
            df["period"] = df["timestamp"].dt.date
            
        # Aggregate metrics by period
        agg_df = df.groupby("period").agg({
            "response_time": "mean",
            "input_tokens": "sum",
            "output_tokens": "sum",
            "total_tokens": "sum",
            "estimated_cost": "sum",
            "success": "mean",
            "is_cached": "mean",
            "timestamp": "count"
        })
        
        # Rename count column to num_queries
        agg_df = agg_df.rename(columns={"timestamp": "num_queries"})
        
        # Reset index to make period a column
        agg_df = agg_df.reset_index()
        
        return agg_df
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison of metrics across different models.
        
        Returns:
            DataFrame with comparative metrics by model
        """
        metrics = self.get_metrics(limit=1000)
        
        if not metrics:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        
        # Aggregate by model_id
        model_df = df.groupby("model_id").agg({
            "response_time": "mean",
            "input_tokens": "mean",
            "output_tokens": "mean", 
            "total_tokens": "mean",
            "estimated_cost": ["mean", "sum"],
            "success": "mean",
            "is_cached": "mean",
            "timestamp": "count"
        })
        
        # Flatten column names
        model_df.columns = ["_".join(col).strip("_") for col in model_df.columns.values]
        
        # Rename count column
        model_df = model_df.rename(columns={"timestamp_count": "num_queries"})
        
        # Reset index
        model_df = model_df.reset_index()
        
        # Calculate cost efficiency (tokens per $)
        model_df["tokens_per_dollar"] = model_df["total_tokens_mean"] / model_df["estimated_cost_mean"]
        
        # Calculate tokens per second
        model_df["tokens_per_second"] = model_df["total_tokens_mean"] / model_df["response_time_mean"]
        
        return model_df
    
    def get_cost_breakdown(self, days: int = 30) -> Dict[str, Any]:
        """
        Get a detailed breakdown of costs by time period, model, and query type.
        
        Args:
            days: Number of days to include in the analysis
            
        Returns:
            Dict with cost breakdown metrics
        """
        metrics = self.get_metrics(limit=10000)
        
        if not metrics:
            return {
                "total_cost": 0,
                "daily_costs": [],
                "model_costs": {},
                "query_type_costs": {},
                "cost_by_date": []
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by recent days
        if days:
            start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= start_date]
        
        # Calculate total cost
        total_cost = df["estimated_cost"].sum()
        
        # Daily cost calculation
        df["date"] = df["timestamp"].dt.date
        daily_costs = df.groupby("date")["estimated_cost"].sum().reset_index()
        daily_costs = daily_costs.sort_values("date")
        
        # Cost by model
        model_costs = df.groupby("model_id")["estimated_cost"].sum().to_dict()
        
        # Cost by query type
        query_type_costs = df.groupby("query_type")["estimated_cost"].sum().to_dict()
        
        # Cost by date and model (for stacked chart)
        cost_by_date_model = df.groupby(["date", "model_id"])["estimated_cost"].sum().reset_index()
        
        # Convert to format suitable for visualization
        cost_by_date = []
        for date, group in cost_by_date_model.groupby("date"):
            date_data = {"date": date.strftime("%Y-%m-%d")}
            for _, row in group.iterrows():
                date_data[row["model_id"]] = row["estimated_cost"]
            cost_by_date.append(date_data)
        
        return {
            "total_cost": total_cost,
            "daily_costs": daily_costs.to_dict(orient="records"),
            "model_costs": model_costs,
            "query_type_costs": query_type_costs,
            "cost_by_date": cost_by_date
        }
    
    def get_template_performance(self) -> pd.DataFrame:
        """
        Analyze the performance of different chat templates.
        
        Returns:
            DataFrame with template performance metrics
        """
        metrics = self.get_metrics(limit=10000)
        
        if not metrics:
            return pd.DataFrame()
        
        # Filter metrics that used templates
        template_metrics = [m for m in metrics if m.get("template_id")]
        
        if not template_metrics:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(template_metrics)
        
        # Group by template_id
        template_df = df.groupby("template_id").agg({
            "response_time": "mean",
            "input_tokens": "mean",
            "output_tokens": "mean", 
            "total_tokens": "mean",
            "estimated_cost": ["mean", "sum"],
            "success": "mean",
            "timestamp": "count"
        })
        
        # Flatten column names
        template_df.columns = ["_".join(col).strip("_") for col in template_df.columns.values]
        
        # Rename count column
        template_df = template_df.rename(columns={"timestamp_count": "usage_count"})
        
        # Reset index
        template_df = template_df.reset_index()
        
        # Calculate efficiency metrics
        template_df["tokens_per_dollar"] = template_df["total_tokens_mean"] / template_df["estimated_cost_mean"]
        template_df["tokens_per_second"] = template_df["total_tokens_mean"] / template_df["response_time_mean"]
        
        return template_df
# Create a singleton instance
metrics_tracker = ModelMetricsTracker()
