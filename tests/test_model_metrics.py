"""
Tests for the model metrics module.
"""

import pytest
import os
import json
import time
import datetime
import pandas as pd
from unittest.mock import patch, MagicMock

from src.model_metrics import ModelMetricsTracker, metrics_tracker

class TestModelMetricsTracker:
    
    @pytest.fixture
    def temp_metrics_file(self, tmpdir):
        """Fixture to create a temporary metrics file for testing."""
        return os.path.join(tmpdir, "test_metrics.json")
    
    def test_initialization(self, temp_metrics_file):
        """Test that the tracker initializes correctly and creates the metrics file."""
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        assert os.path.exists(temp_metrics_file)
        
        # Check that the file contains an empty list
        with open(temp_metrics_file, 'r') as f:
            data = json.load(f)
            assert data == []
    
    def test_start_tracking(self):
        """Test that start_tracking returns a timestamp."""
        tracker = ModelMetricsTracker()
        start_time = tracker.start_tracking()
        
        # Verify it's a recent timestamp (within the last second)
        current_time = time.time()
        assert start_time <= current_time
        assert start_time > current_time - 1
    
    def test_record_metrics(self, temp_metrics_file):
        """Test recording metrics to storage."""
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        start_time = time.time() - 0.5  # Simulate a 500ms API call
        
        metrics = tracker.record_metrics(
            model_id="llama-3.1-70b",
            start_time=start_time,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            is_cached=False,
            query_type="text",
            success=True
        )
        
        # Verify returned metrics
        assert metrics["model_id"] == "llama-3.1-70b"
        assert metrics["input_tokens"] == 100
        assert metrics["output_tokens"] == 200
        assert metrics["total_tokens"] == 300
        assert 0.4 < metrics["response_time"] < 0.6  # Around 500ms
        assert metrics["is_cached"] == False
        assert metrics["success"] == True
        
        # Verify metrics were saved to file
        with open(temp_metrics_file, 'r') as f:
            saved_metrics = json.load(f)
            assert len(saved_metrics) == 1
            assert saved_metrics[0]["model_id"] == "llama-3.1-70b"
    
    def test_calculate_cost(self):
        """Test that cost calculation is correct."""
        tracker = ModelMetricsTracker()
        
        # Test known model cost
        cost = tracker._calculate_cost("llama-3.1-70b", 1000, 500)
        expected_cost = (1000 * 0.00000015) + (500 * 0.00000045)
        assert cost == expected_cost
        
        # Test unknown model (should use default cost)
        cost = tracker._calculate_cost("unknown-model", 1000, 500)
        expected_cost = (1000 * 0.00000020) + (500 * 0.00000060)
        assert cost == expected_cost
    
    def test_get_metrics(self, temp_metrics_file):
        """Test retrieving metrics with filtering."""
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        
        # Add some test metrics
        base_time = datetime.datetime(2025, 5, 1, 12, 0, 0).isoformat()
        test_metrics = [
            {
                "timestamp": datetime.datetime(2025, 5, 1, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.5,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 2, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-8b",
                "response_time": 0.3,
                "input_tokens": 100,
                "output_tokens": 150,
                "total_tokens": 250,
                "estimated_cost": 0.0001,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 3, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.7,
                "input_tokens": 120,
                "output_tokens": 180,
                "total_tokens": 300,
                "estimated_cost": 0.00018,
                "is_cached": True,
                "query_type": "text",
                "success": True,
                "error_type": None
            }
        ]
        
        with open(temp_metrics_file, 'w') as f:
            json.dump(test_metrics, f)
        
        # Test getting all metrics
        all_metrics = tracker.get_metrics()
        assert len(all_metrics) == 3
        
        # Test model filter
        filtered_metrics = tracker.get_metrics(model_filter="llama-3.1-70b")
        assert len(filtered_metrics) == 2
        assert all(m["model_id"] == "llama-3.1-70b" for m in filtered_metrics)
        
        # Test date filter
        date_from = datetime.datetime(2025, 5, 2)
        filtered_metrics = tracker.get_metrics(date_from=date_from)
        assert len(filtered_metrics) == 2
        assert all(m["timestamp"] >= date_from.isoformat() for m in filtered_metrics)
    
    def test_get_metrics_summary(self, temp_metrics_file):
        """Test that metrics summary is calculated correctly."""
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        
        # Add some test metrics
        test_metrics = [
            {
                "timestamp": datetime.datetime(2025, 5, 1, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.5,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 2, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-8b",
                "response_time": 0.3,
                "input_tokens": 100,
                "output_tokens": 150,
                "total_tokens": 250,
                "estimated_cost": 0.0001,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 3, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.7,
                "input_tokens": 120,
                "output_tokens": 180,
                "total_tokens": 300,
                "estimated_cost": 0.00018,
                "is_cached": True,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 4, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.6,
                "input_tokens": 150,
                "output_tokens": 250,
                "total_tokens": 400,
                "estimated_cost": 0.00025,
                "is_cached": False,
                "query_type": "text",
                "success": False,
                "error_type": "APIError"
            }
        ]
        
        with open(temp_metrics_file, 'w') as f:
            json.dump(test_metrics, f)
        
        summary = tracker.get_metrics_summary()
        
        # Test summary calculations
        assert summary["total_queries"] == 4
        assert summary["total_tokens"] == 1250
        assert round(summary["total_cost"], 5) == 0.00073  # Use rounded comparison due to floating-point precision
        assert summary["success_rate"] == 75.0
        assert summary["cache_hit_rate"] == 25.0
        
        # Test model stats
        assert len(summary["model_stats"]) == 2
        assert summary["model_stats"]["llama-3.1-70b"]["count"] == 3
        assert summary["model_stats"]["llama-3.1-8b"]["count"] == 1
    
    @patch('streamlit.metric')
    @patch('streamlit.plotly_chart')
    def test_visualize_metrics_with_data(self, mock_plotly, mock_metric, temp_metrics_file):
        """Test that metrics visualization works with data."""
        # This is a basic test to check function runs without errors
        # Full visualization testing would be more complex
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        
        # Add some test metrics
        test_metrics = [
            {
                "timestamp": datetime.datetime(2025, 5, 1, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.5,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            }
        ]
        
        with open(temp_metrics_file, 'w') as f:
            json.dump(test_metrics, f)
        
        # Create mock columns
        mock_col = MagicMock()
        mock_cols = MagicMock(return_value=[mock_col, mock_col, mock_col, mock_col])
        
        # Create mock tabs
        mock_tab = MagicMock()
        mock_tabs = MagicMock(return_value=[mock_tab, mock_tab, mock_tab])
        
        # Patch methods that interact with streamlit
        with patch('streamlit.header'), \
             patch('streamlit.subheader'), \
             patch('streamlit.columns', mock_cols), \
             patch('streamlit.tabs', mock_tabs), \
             patch('streamlit.dataframe'), \
             patch('streamlit.info'), \
             patch('pandas.DataFrame.groupby'), \
             patch('plotly.express.pie'), \
             patch('plotly.express.line'), \
             patch('plotly.express.bar'):
            
            # Skip actual visualization testing as it's complex to mock
            # Just ensure the function doesn't raise exceptions
            try:
                tracker.visualize_metrics()
                visualization_succeeded = True
            except Exception:
                visualization_succeeded = False
            
            assert visualization_succeeded
    
    def test_get_metrics_by_time_period(self, temp_metrics_file):
        """Test metrics aggregation by time period"""
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        
        # Add test metrics spanning multiple days
        test_metrics = [
            # Day 1: May 1
            {
                "timestamp": datetime.datetime(2025, 5, 1, 9, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.5,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 1, 14, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.6,
                "input_tokens": 110,
                "output_tokens": 190,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            # Day 2: May 2
            {
                "timestamp": datetime.datetime(2025, 5, 2, 10, 0, 0).isoformat(),
                "model_id": "llama-3.1-8b",
                "response_time": 0.3,
                "input_tokens": 100,
                "output_tokens": 150,
                "total_tokens": 250,
                "estimated_cost": 0.0001,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            # Day 3: May 3
            {
                "timestamp": datetime.datetime(2025, 5, 3, 11, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.7,
                "input_tokens": 120,
                "output_tokens": 180,
                "total_tokens": 300,
                "estimated_cost": 0.00018,
                "is_cached": True,
                "query_type": "vision",
                "success": True,
                "error_type": None
            },
        ]
        
        with open(temp_metrics_file, 'w') as f:
            json.dump(test_metrics, f)
            
        # Test daily aggregation
        daily_metrics = tracker.get_metrics_by_time_period(period_type="day")
        assert len(daily_metrics) == 3  # 3 days
        
        # Find May 1 data (should have 2 queries)
        may1_date = pd.Timestamp("2025-05-01").date()
        may_1_rows = daily_metrics[daily_metrics["period"] == may1_date]
        assert len(may_1_rows) == 1
        assert may_1_rows.iloc[0]["num_queries"] == 2
        assert may_1_rows.iloc[0]["total_tokens"] == 600
        
        # Test model filter
        filtered_metrics = tracker.get_metrics_by_time_period(
            period_type="day", 
            model_filter="llama-3.1-70b"
        )
        assert len(filtered_metrics) == 2  # Only May 1 and May 3 have this model
        
        # Test day limit
        limited_metrics = tracker.get_metrics_by_time_period(
            period_type="day",
            limit_days=2
        )
        assert len(limited_metrics) <= 2
    
    def test_get_model_comparison(self, temp_metrics_file):
        """Test model comparison analytics"""
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        
        # Add test metrics for different models
        test_metrics = [
            # Model 1: llama-3.1-70b (3 queries)
            {
                "timestamp": datetime.datetime(2025, 5, 1, 9, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.5,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 2, 10, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.6,
                "input_tokens": 110,
                "output_tokens": 190,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 3, 11, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.7,
                "input_tokens": 120,
                "output_tokens": 180,
                "total_tokens": 300,
                "estimated_cost": 0.00018,
                "is_cached": True,
                "query_type": "text",
                "success": False,
                "error_type": "APIError"
            },
            # Model 2: llama-3.1-8b (2 queries)
            {
                "timestamp": datetime.datetime(2025, 5, 1, 12, 0, 0).isoformat(),
                "model_id": "llama-3.1-8b",
                "response_time": 0.3,
                "input_tokens": 100,
                "output_tokens": 150,
                "total_tokens": 250,
                "estimated_cost": 0.0001,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            {
                "timestamp": datetime.datetime(2025, 5, 2, 14, 0, 0).isoformat(),
                "model_id": "llama-3.1-8b",
                "response_time": 0.35,
                "input_tokens": 90,
                "output_tokens": 160,
                "total_tokens": 250,
                "estimated_cost": 0.0001,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
        ]
        
        with open(temp_metrics_file, 'w') as f:
            json.dump(test_metrics, f)
            
        # Get model comparison
        comparison = tracker.get_model_comparison()
        
        # Verify we have 2 models
        assert len(comparison) == 2
        
        # Get llama-3.1-70b data
        llama_70b_rows = comparison[comparison["model_id"] == "llama-3.1-70b"]
        llama_8b_rows = comparison[comparison["model_id"] == "llama-3.1-8b"]
        
        assert len(llama_70b_rows) == 1
        assert len(llama_8b_rows) == 1
        
        llama_70b = llama_70b_rows.iloc[0]
        llama_8b = llama_8b_rows.iloc[0]
        
        # Check metrics
        assert llama_70b["num_queries"] == 3
        assert llama_8b["num_queries"] == 2
        
        # Check calculated efficiency metrics
        assert llama_70b["tokens_per_dollar"] > 0
        assert llama_70b["tokens_per_second"] > 0
        assert llama_8b["tokens_per_dollar"] > 0
        assert llama_8b["tokens_per_second"] > 0
        
        # Verify success rates
        assert round(llama_70b["success_mean"], 2) == round(2/3, 2)  # 2 out of 3 successful
        assert round(llama_8b["success_mean"], 2) == 1.0  # All successful
        
        # Verify llama-8b should be faster than llama-70b
        assert llama_8b["response_time_mean"] < llama_70b["response_time_mean"]
    
    def test_get_cost_breakdown(self, temp_metrics_file):
        """Test cost breakdown analytics"""
        tracker = ModelMetricsTracker(storage_path=temp_metrics_file)
        
        # Add test metrics over multiple days with different costs
        test_metrics = [
            # Day 1: May 1
            {
                "timestamp": datetime.datetime(2025, 5, 1, 9, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b",
                "response_time": 0.5,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "estimated_cost": 0.0002,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            # Day 2: May 2
            {
                "timestamp": datetime.datetime(2025, 5, 2, 10, 0, 0).isoformat(),
                "model_id": "llama-3.1-8b",
                "response_time": 0.3,
                "input_tokens": 100,
                "output_tokens": 150,
                "total_tokens": 250,
                "estimated_cost": 0.0001,
                "is_cached": False,
                "query_type": "text",
                "success": True,
                "error_type": None
            },
            # Day 3: May 3
            {
                "timestamp": datetime.datetime(2025, 5, 3, 11, 0, 0).isoformat(),
                "model_id": "llama-3.1-70b-vision",
                "response_time": 0.7,
                "input_tokens": 1200,
                "output_tokens": 180,
                "total_tokens": 1380,
                "estimated_cost": 0.0005,
                "is_cached": False,
                "query_type": "vision",
                "success": True,
                "error_type": None
            },
        ]
        
        with open(temp_metrics_file, 'w') as f:
            json.dump(test_metrics, f)
            
        # Get cost breakdown
        cost_data = tracker.get_cost_breakdown(days=30)
        
        # Check total cost
        expected_cost = 0.0002 + 0.0001 + 0.0005
        assert round(cost_data["total_cost"], 6) == round(expected_cost, 6)
        
        # Check daily costs
        assert len(cost_data["daily_costs"]) == 3
        
        # Check model costs
        assert len(cost_data["model_costs"]) == 3
        assert round(cost_data["model_costs"]["llama-3.1-70b"], 6) == 0.0002
        assert round(cost_data["model_costs"]["llama-3.1-8b"], 6) == 0.0001
        assert round(cost_data["model_costs"]["llama-3.1-70b-vision"], 6) == 0.0005
        
        # Check query type costs
        assert len(cost_data["query_type_costs"]) == 2
        assert round(cost_data["query_type_costs"]["text"], 6) == 0.0003
        assert round(cost_data["query_type_costs"]["vision"], 6) == 0.0005
        
        # Test with limited days
        limited_cost_data = tracker.get_cost_breakdown(days=1)
        assert limited_cost_data["total_cost"] <= 0.0005  # Should only include most recent day
        
        # Check cost_by_date
        assert len(cost_data["cost_by_date"]) > 0
