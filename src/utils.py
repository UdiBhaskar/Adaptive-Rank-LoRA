"""
Utility Functions for Adaptive Rank LoRA Training

This module provides utility functions for data loading, logging setup, system monitoring,
and configuration management used throughout the Adaptive Rank LoRA training pipeline.

Key Components:
- Dictionary flattening for configuration serialization
- Dataset loading from Parquet files
- Comprehensive logging setup
- System monitoring callbacks for GPU/CPU/memory tracking
- Performance monitoring for training sessions

The module is designed to support robust training workflows with comprehensive
monitoring and configuration management capabilities.
"""

from collections.abc import Mapping
from datasets import load_dataset
import logging
import time
from transformers import TrainerCallback
import pynvml
import psutil
import os
import random

# Set random seed for reproducible behavior in utility functions
random.seed(24)


def flatten_dict(nested, separator="_"):
    """
    Recursively flattens a nested dictionary by concatenating keys with a separator.
    
    This function is useful for converting nested configuration dictionaries into
    flat key-value pairs suitable for logging, serialization, or storage in
    systems that don't support nested structures.
    
    Args:
        nested (dict): The nested dictionary to flatten
        separator (str): String to use for joining nested keys. Defaults to "_".
        
    Returns:
        dict: A flattened dictionary with concatenated keys
        
    Raises:
        ValueError: If any key contains the separator string
        
    Example:
        >>> nested = {"model": {"lora": {"rank": 16, "alpha": 32}}, "training": {"lr": 0.001}}
        >>> flatten_dict(nested)
        {"model_lora_rank": 16, "model_lora_alpha": 32, "training_lr": 0.001}
    """
    def recurse(nest, prefix, into):
        for key, value in nest.items():
            if separator in key:
                raise ValueError(f"separator '{separator}' not allowed in key '{key}'")
            if isinstance(value, Mapping):
                recurse(value, prefix + key + separator, into)
            else:
                into[prefix + key] = value

    flattened = {}
    recurse(nested, "", flattened)
    return flattened



def load_raw_data(data_args):
    """
    Loads training and validation datasets from Parquet files.
    
    This function creates a HuggingFace dataset dictionary from Parquet files
    specified in the data arguments. Parquet format is preferred for its
    efficiency and support for columnar data.
    
    Args:
        data_args: DataArguments object containing:
            - train_dataset_path: Path to training data Parquet file
            - val_dataset_path: Path to validation data Parquet file
    
    Returns:
        datasets.DatasetDict: Dictionary containing 'train' and 'val' datasets
        
    Raises:
        FileNotFoundError: If specified Parquet files don't exist
        ValueError: If Parquet files are malformed or incompatible
    """
    data_files = {
        "train": data_args.train_dataset_path,
        "val": data_args.val_dataset_path
    }
    raw_data = load_dataset("parquet", data_files=data_files)
    return raw_data


def setup_logging(log_file_path):
    """
    Sets up comprehensive logging with both file and console output.
    
    This function configures a logger that writes to both a file and the console,
    making it easy to monitor training progress and debug issues. The logging
    format includes timestamps, module names, and log levels for easy parsing.
    
    Args:
        log_file_path (str): Path where the log file should be created
        
    Returns:
        logging.Logger: Configured logger instance ready for use
        
    Features:
        - Dual output: both file and console logging
        - Timestamped entries for temporal analysis
        - INFO level logging (captures important events without spam)
        - Standardized format for consistency
    """
    # Create a logger instance with module-specific naming
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Capture INFO level and above

    # Create file handler for persistent logging
    file_handler = logging.FileHandler(log_file_path)
    
    # Create console handler for real-time monitoring
    console_handler = logging.StreamHandler()

    # Define comprehensive logging format with timestamp and context
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Apply consistent formatting to both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Register handlers with the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class SystemMonitorCallback(TrainerCallback):
    """
    Comprehensive system monitoring callback for training sessions.
    
    This callback monitors and logs system resources (CPU, GPU, memory) during
    training to help identify bottlenecks, resource utilization patterns, and
    potential issues. The metrics are logged to TensorBoard for visualization.
    
    Features:
        - Real-time GPU utilization and memory tracking
        - CPU usage and frequency monitoring  
        - System memory (RAM) usage tracking
        - GPU temperature and power consumption monitoring
        - TensorBoard integration for metric visualization
        - Configurable logging intervals to balance overhead vs. granularity
    
    Attributes:
        logging_interval (float): Time interval between metric collections (seconds)
        last_log_time (float): Timestamp of last metric collection
        logger: Logger instance for status messages
        device_count (int): Number of available GPU devices
        handles (list): NVML device handles for GPU monitoring
        tb_writer: TensorBoard SummaryWriter instance
    """

    def __init__(self, logger, logging_interval=1):
        """
        Initialize the system monitoring callback.
        
        Args:
            logger: Logger instance for status messages and warnings
            logging_interval (float): Interval between metric collections in seconds.
                Lower values provide more granular data but may impact performance.
        """
        self.logging_interval = logging_interval
        self.last_log_time = 0
        self.logger = logger

        # Initialize NVIDIA Management Library (NVML) for GPU monitoring
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)
        ]

        # Try to import TensorBoard SummaryWriter with fallback options
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
            self._SummaryWriter = SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
                self._SummaryWriter = SummaryWriter
            except ImportError:
                self._SummaryWriter = None
                self.logger.warning("TensorBoard not available - metrics will not be logged")

    def _get_system_metrics(self):
        """
        Collects comprehensive system performance metrics.
        
        This method gathers real-time information about system resource usage
        including CPU, memory, and GPU metrics. The data is structured for
        easy logging and visualization in monitoring systems.
        
        Returns:
            dict: Nested dictionary containing:
                - cpu: CPU usage percentage and frequency
                - memory: RAM usage statistics in GB and percentages
                - gpu: List of per-GPU metrics including utilization, memory, temp, power
                
        GPU Metrics per Device:
            - id: GPU device identifier
            - utilization: GPU core utilization percentage (0-100)
            - memory_used_mb/memory_free_mb: VRAM usage in megabytes
            - memory_percent: VRAM usage as percentage of total
            - temperature: GPU temperature in Celsius
            - power_watts: Current power consumption in watts
        """
        metrics = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=None),  # Current CPU usage
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,  # CPU frequency in MHz
            },
            "memory": {
                "percent": psutil.virtual_memory().percent,  # Memory usage percentage
                "used_gb": psutil.virtual_memory().used / (1024**3),  # Used memory in GB
                "available_gb": psutil.virtual_memory().available / (1024**3),  # Available memory in GB
            },
            "gpu": [],  # Will be populated with per-GPU metrics
        }

        # Collect metrics for each available GPU device
        for device_index, gpu_handle in enumerate(self.handles):
            try:
                # Get GPU utilization rates (GPU and memory utilization)
                utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                
                # Get detailed memory information
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                
                # Get GPU temperature
                temperature = pynvml.nvmlDeviceGetTemperature(
                    gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Get power consumption (convert from milliwatts to watts)
                power_consumption = (
                    pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
                )

                # Compile GPU metrics for this device
                gpu_metrics = {
                    "id": device_index,
                    "utilization": utilization_rates.gpu,  # GPU core utilization %
                    "memory_used_mb": memory_info.used / (1024**2),  # VRAM used in MB
                    "memory_free_mb": memory_info.free / (1024**2),  # VRAM free in MB
                    "memory_percent": (memory_info.used / memory_info.total) * 100,  # VRAM usage %
                    "temperature": temperature,  # Temperature in Celsius
                    "power_watts": power_consumption,  # Power usage in watts
                }
                
                metrics["gpu"].append(gpu_metrics)
                
            except pynvml.NVMLError as e:
                # Log GPU monitoring errors but continue with other devices
                self.logger.warning(f"Error collecting metrics for GPU {device_index}: {e}")

        return metrics

    def _log_to_tensorboard(self, system_metrics, state):
        """
        Logs collected system metrics to TensorBoard for visualization.
        
        This method writes the collected system metrics to TensorBoard using
        structured naming conventions that make it easy to create dashboards
        and monitor training progress alongside system resource usage.
        
        Args:
            system_metrics (dict): Metrics collected from _get_system_metrics()
            state: Trainer state object containing global_step for timestamping
            
        Metric Organization in TensorBoard:
            - cpu/percent, cpu/frequency: CPU-related metrics
            - memory/percent, memory/used_gb, memory/available_gb: Memory metrics
            - utilization/gpu_X, memory_percent/gpu_X, etc.: Per-GPU metrics
        """
        # Skip logging if TensorBoard is not available or state is invalid
        if not self.tb_writer or not hasattr(state, "global_step"):
            return

        # Log CPU metrics with consistent naming
        for metric_name, value in system_metrics["cpu"].items():
            self.tb_writer.add_scalar(f"cpu/{metric_name}", value, state.global_step)

        # Log memory metrics
        for metric_name, value in system_metrics["memory"].items():
            self.tb_writer.add_scalar(f"memory/{metric_name}", value, state.global_step)

        # Log GPU metrics with per-device identification
        for gpu_data in system_metrics["gpu"]:
            gpu_id = gpu_data["id"]
            for metric_name, value in gpu_data.items():
                if metric_name != "id":  # Skip the ID field itself
                    self.tb_writer.add_scalar(
                        f"{metric_name}/gpu_{gpu_id}", value, state.global_step
                    )
        
        # Ensure metrics are written to disk immediately
        self.tb_writer.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Initialize monitoring infrastructure when training begins.
        
        This callback method sets up TensorBoard logging and initializes
        the monitoring system. It creates a dedicated subdirectory for
        system monitoring logs within the training output directory.
        
        Args:
            args: Training arguments containing output directory
            state: Training state (not used but required by callback interface)
            control: Training control (not used but required by callback interface)
            **kwargs: Additional keyword arguments from the trainer
        """
        if self._SummaryWriter is not None:
            # Create dedicated directory for system monitoring logs
            log_dir = os.path.join(args.output_dir, "system_monitoring")
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)
            self.logger.info(f"System monitoring initialized - TensorBoard logs: {log_dir}")
        else:
            self.tb_writer = None
            self.logger.warning("TensorBoard not available - system metrics will not be logged")

    def on_step_end(self, args, state, control, **kwargs):
        """
        Collect and log system metrics at the end of each training step.
        
        This method implements the main monitoring loop, collecting metrics
        at regular intervals to avoid excessive overhead while maintaining
        useful temporal resolution for analysis.
        
        Args:
            args: Training arguments (not used but required by callback interface)
            state: Training state containing step information
            control: Training control (not used but required by callback interface)
            **kwargs: Additional keyword arguments from the trainer
        """
        current_time = time.time()

        # Check if enough time has passed since last collection
        if current_time - self.last_log_time >= self.logging_interval:
            # Collect current system metrics
            system_metrics = self._get_system_metrics()

            # Log metrics to TensorBoard for visualization
            self._log_to_tensorboard(system_metrics, state)

            # Update timestamp for next collection interval
            self.last_log_time = current_time

    def on_train_end(self, args, state, control, **kwargs):
        """
        Clean up monitoring resources when training ends.
        
        This method properly shuts down NVML monitoring and closes the
        TensorBoard writer to ensure all data is saved and resources
        are freed properly.
        
        Args:
            args: Training arguments (not used but required by callback interface)
            state: Training state (not used but required by callback interface)  
            control: Training control (not used but required by callback interface)
            **kwargs: Additional keyword arguments from the trainer
        """
        # Shutdown NVIDIA monitoring library
        pynvml.nvmlShutdown()
        
        # Close TensorBoard writer and free resources
        if hasattr(self, 'tb_writer') and self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
            self.logger.info("System monitoring stopped and resources cleaned up")


class BatchTimeCallback(TrainerCallback):
    """
    Callback to track and log batch processing times during training.
    
    This callback provides insights into training performance by measuring
    the time taken to process each batch. This information is crucial for
    identifying performance bottlenecks and optimizing training efficiency.
    
    Features:
        - Per-batch timing measurement
        - Average batch time calculation
        - Throughput estimation (samples/tokens per second)
        - Integration with existing logging infrastructure
    
    Attributes:
        logger: Logger instance for output
        start_time (float): Timestamp when current batch started
        batch_times (list): History of recent batch processing times
        step_count (int): Number of completed training steps
    """
    
    def __init__(self, logger, window_size=100):
        """
        Initialize the batch timing callback.
        
        Args:
            logger: Logger instance for outputting timing information
            window_size (int): Number of recent batch times to keep for averaging.
                Larger windows provide more stable averages but use more memory.
        """
        self.logger = logger
        self.start_time = None
        self.batch_times = []
        self.window_size = window_size
        self.step_count = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        """
        Record the start time of each training step.
        
        Args:
            args: Training arguments (not used but required by callback interface)
            state: Training state (not used but required by callback interface)
            control: Training control (not used but required by callback interface) 
            **kwargs: Additional keyword arguments from the trainer
        """
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Calculate and log batch processing time.
        
        This method computes the time taken for the current batch and
        maintains running statistics for performance analysis.
        
        Args:
            args: Training arguments containing batch size information
            state: Training state containing step information
            control: Training control (not used but required by callback interface)
            **kwargs: Additional keyword arguments from the trainer
        """
        if self.start_time is not None:
            # Calculate time for this batch
            batch_time = time.time() - self.start_time
            self.batch_times.append(batch_time)
            self.step_count += 1
            
            # Maintain sliding window of batch times
            if len(self.batch_times) > self.window_size:
                self.batch_times.pop(0)
            
            # Log timing information periodically
            if self.step_count % 50 == 0:  # Log every 50 steps
                avg_batch_time = sum(self.batch_times) / len(self.batch_times)
                
                # Calculate throughput if batch size is available
                if hasattr(args, 'per_device_train_batch_size'):
                    samples_per_second = args.per_device_train_batch_size / avg_batch_time
                    self.logger.info(
                        f"Step {state.global_step}: Avg batch time: {avg_batch_time:.3f}s, "
                        f"Throughput: {samples_per_second:.1f} samples/sec"
                    )
                else:
                    self.logger.info(
                        f"Step {state.global_step}: Average batch time: {avg_batch_time:.3f}s"
                    )
            
            # Reset for next batch
            self.start_time = None
