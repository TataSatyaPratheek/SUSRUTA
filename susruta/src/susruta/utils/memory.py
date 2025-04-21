"""Memory tracking utilities for monitoring and optimizing memory usage."""

import gc
import psutil
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class MemoryTracker:
    """Utility class to track memory usage during processing."""
    
    def __init__(self, threshold_mb: float = 7000, process: Optional[psutil.Process] = None):
        """
        Initialize memory tracker.
        
        Args:
            threshold_mb: Memory threshold in MB for warnings
            process: Specific process to track, defaults to current process
        """
        self.process = process or psutil.Process()
        self.threshold_mb = threshold_mb
        self.log: List[Tuple[str, float]] = []
    
    def current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def log_memory(self, operation: str) -> float:
        """
        Log memory usage for an operation.
        
        Args:
            operation: Name of the operation being performed
            
        Returns:
            Current memory usage in MB
        """
        memory_mb = self.current_memory_mb()
        self.log.append((operation, memory_mb))
        if memory_mb > self.threshold_mb:
            print(f"WARNING: {operation} used {memory_mb:.2f}MB, exceeding threshold of {self.threshold_mb}MB")
        return memory_mb
    
    def plot_memory_usage(self) -> plt.Figure:
        """
        Plot memory usage over time.
        
        Returns:
            Matplotlib figure with memory usage plot
        """
        operations, memory = zip(*self.log)
        fig = plt.figure(figsize=(12, 6))
        plt.plot(memory)
        plt.xticks(range(len(operations)), operations, rotation=90)
        plt.xlabel('Operation')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage During Processing')
        plt.tight_layout()
        return fig
    
    def free_memory(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        self.log_memory("After garbage collection")