# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import os
import omegaconf
from dotenv import load_dotenv

load_dotenv()

ENV_VAR_NOT_FOUND_ERR = (
    "{env_var_name} environment variable is not set or is empty."
    " Make sure to set it in your .env file or in the environment variables."
)


def get_hardware():
    """Determine available hardware (GPU or CPU)."""
    try:
        # Check if `nvidia-smi` is available and get GPU name
        result = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True)
        # Process output: trim spaces, remove duplicates, and format
        hardware = ", ".join(sorted(set(line.strip() for line in result.split("\n") if line.strip())))
    except subprocess.CalledProcessError:
        hardware = "a CPU"  # Default if no GPU is found
    return hardware


def check_pytorch_gpu():
    """Check if PyTorch can use a GPU."""
    try:
        return subprocess.check_output(
            "python -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')\"",
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "ERROR: PyTorch check failed"


def check_tensorflow_gpu():
    """Check if TensorFlow can use a GPU."""
    try:
        return subprocess.check_output(
            "python -c \"import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))\"",
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "ERROR: TensorFlow check failed"


def format_time(seconds):
    """Convert time in seconds to a human-readable format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}hrs {minutes}mins {secs}secs"


def get_log_dir():
    """Get the log directory, creating it if it doesn't exist."""
    log_dir = os.getenv("LOGGING_DIR", "")
    if not log_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="LOGGING_DIR"))
    return log_dir


def get_superimage_dir():
    """Get the superimage directory, creating it if it doesn't exist."""
    superimage_dir = os.getenv("SUPERIMAGE_DIR", "")
    # raise an error if empty string or None
    if not superimage_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="SUPERIMAGE_DIR"))
    return superimage_dir


def get_mlebench_data_dir():
    """Get the MLEBench data directory, creating it if it doesn't exist."""
    mlebench_data_dir = os.getenv("MLE_BENCH_DATA_DIR", "")
    if not mlebench_data_dir:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="MLE_BENCH_DATA_DIR"))
    return mlebench_data_dir


def get_default_slurm_partition():
    """Get the default Slurm partition from the configuration."""
    slurm_partition = os.getenv("DEFAULT_SLURM_PARTITION","")
    if not slurm_partition:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_PARTITION"))
    return slurm_partition


def get_default_slurm_account():
    """Get the default Slurm account from the configuration."""
    slurm_account = os.getenv("DEFAULT_SLURM_ACCOUNT", "")
    if not slurm_account:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_ACCOUNT"))
    return slurm_account


def get_default_slurm_qos():
    """Get the default Slurm QoS from the configuration."""
    slurm_qos = os.getenv("DEFAULT_SLURM_QOS", "")
    if not slurm_qos:
        raise ValueError(ENV_VAR_NOT_FOUND_ERR.format(env_var_name="DEFAULT_SLURM_QOS"))
    return slurm_qos
