#!/bin/bash

# Enhanced memory configuration for H100 (80GB)
export XLA_FLAGS="--xla_gpu_enable_command_buffer='' \
  --xla_gpu_enable_custom_fusions=false \
  --xla_gpu_enable_while_loop_double_buffering=false \
  --xla_gpu_deterministic_ops=true \
  --xla_force_host_platform_device_count=1 \
  --xla_gpu_memory_limit_slop_factor=95"

# JAX-specific optimizations
export JAX_ENABLE_X64=0  # Use float32 to save memory
export JAX_DEFAULT_MATMUL_PRECISION="high"
export JAX_TRACEBACK_FILTERING=off  # Better debugging

# Unified memory with larger fraction
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
