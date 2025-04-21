"""Tests for the utilities module."""

import pytest
import gc
import psutil
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from susruta.utils import MemoryTracker


class TestMemoryTracker:
    """Test suite for the MemoryTracker class."""
    
    def test_initialization(self):
        """Test initialization of MemoryTracker."""
        # Test with default threshold
        tracker = MemoryTracker()
        assert tracker.threshold_mb == 7000
        assert isinstance(tracker.process, psutil.Process)
        assert hasattr(tracker, 'log')
        assert isinstance(tracker.log, list)
        assert len(tracker.log) == 0
        
        # Test with custom threshold
        tracker = MemoryTracker(threshold_mb=5000)
        assert tracker.threshold_mb == 5000
        
        # Test with custom process
        mock_process = MagicMock()
        tracker = MemoryTracker(process=mock_process)
        assert tracker.process == mock_process
    
    @patch('psutil.Process')
    def test_current_memory_mb(self, mock_process_class):
        """Test getting current memory usage."""
        # Setup mock
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process
        mock_process.memory_info.return_value.rss = 500 * 1024 * 1024  # 500 MB in bytes
        
        # Create tracker with the mock process
        tracker = MemoryTracker()
        
        # Test getting memory
        memory_mb = tracker.current_memory_mb()
        
        assert memory_mb == 500.0
        mock_process.memory_info.assert_called_once()
    
    @patch('psutil.Process')
    def test_log_memory(self, mock_process_class):
        """Test logging memory usage."""
        # Setup mock
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process
        mock_process.memory_info.return_value.rss = 500 * 1024 * 1024  # 500 MB in bytes
        
        # Create tracker with the mock process
        tracker = MemoryTracker(threshold_mb=1000)
        
        # Test logging memory - below threshold
        memory_mb = tracker.log_memory("Operation 1")
        
        assert memory_mb == 500.0
        assert len(tracker.log) == 1
        assert tracker.log[0] == ("Operation 1", 500.0)
        
        # Test logging memory - above threshold
        mock_process.memory_info.return_value.rss = 1500 * 1024 * 1024  # 1500 MB
        
        with patch('builtins.print') as mock_print:
            memory_mb = tracker.log_memory("Operation 2")
            
            assert memory_mb == 1500.0
            assert len(tracker.log) == 2
            assert tracker.log[1] == ("Operation 2", 1500.0)
            
            # Check warning was printed
            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert "WARNING" in args
            assert "Operation 2" in args
            assert "1500.00MB" in args
            assert "1000MB" in args
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_memory_usage(self, mock_tight_layout, mock_title, mock_ylabel, 
                               mock_xlabel, mock_xticks, mock_plot, mock_figure):
        """Test plotting memory usage."""
        # Setup mock
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Create tracker with some log entries
        tracker = MemoryTracker()
        tracker.log = [
            ("Operation 1", 500.0),
            ("Operation 2", 600.0),
            ("Operation 3", 550.0)
        ]
        
        # Test plotting
        fig = tracker.plot_memory_usage()
        
        # Check that all plotting functions were called
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        mock_xticks.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_title.assert_called_once()
        mock_tight_layout.assert_called_once()
        
        # Check that figure was returned
        assert fig == mock_fig
    
    @patch('gc.collect')
    def test_free_memory(self, mock_gc_collect):
        """Test memory freeing."""
        # Create tracker with mock log_memory method
        tracker = MemoryTracker()
        tracker.log_memory = MagicMock(return_value=500.0)
        
        # Test freeing memory
        tracker.free_memory()
        
        mock_gc_collect.assert_called_once()
        tracker.log_memory.assert_called_once_with("After garbage collection")