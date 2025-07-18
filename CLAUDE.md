# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `mldistill`, a machine learning model distillation framework focused on knowledge distillation for language models. The project provides tools for training student models from teacher models using various sampling strategies and distillation techniques.

## Key Architecture Components

### Core Framework Structure
- **`mldistill/`** - Main package containing the distillation framework
  - `offpolicy.py` - Core distillation logic and main `distill()` function
  - `train.py` - `Trainer` class handling the training loop, validation, and checkpointing
  - `standard.py` - CLI entry point for standard distillation with individual data files
  - `regmix.py` - CLI entry point for mixture-based distillation using proportional sampling
  - `sampler.py` - Data sampling strategies (`RandomSampler`, `ProportionalSampler`)
  - `utils.py` - Utility functions for data loading, logging, checkpointing, etc.

### Distillation Modes
The framework supports two main distillation entry points:
1. **Standard mode** (`mld-standard`): Takes individual parquet files as input
2. **RegMix mode** (`mld-regmix`): Takes a mixture file defining datasets and their proportions

### Training Framework
- Uses HuggingFace Transformers and Accelerate for distributed training
- Supports various Gemma model configurations
- Implements KL divergence + cross-entropy loss for distillation
- Includes gradient checkpointing, optimizer offloading, and memory management

## Common Development Commands

### Installation
```bash
pip install -e .
```

### Running Distillation

**Standard mode (individual files):**
```bash
mld-standard train_data.parquet --val-data-files val_data.parquet --pretrained --distillation
```

**RegMix mode (mixture file):**
```bash
mld-regmix mixture.txt --pretrained --distillation
```

### Test Scripts
Example test scripts are in `test/` directory:
- `test/test.sh` - Main test script with accelerate launch configuration
- `test/dyna_gemma3_1b.sh` - Dynamic dataset test
- `test/giga_gemma3_1b_distill.sh` - Gigaword distillation test

Run tests with:
```bash
./test/test.sh
```

### Key Training Parameters
- `--student`: Student model path (default: `models/gemma-3-1b-pt`)
- `--teacher`: Teacher model path (default: `models/gemma-3-4b-pt`) 
- `--pretrained`: Initialize from pretrained model vs fresh config
- `--distillation`: Enable distillation (otherwise standard training)
- `--alpha`: Weight for KL divergence loss (default: 1.0)
- `--max-seq-length`: Maximum sequence length (default: 4096)
- `--batch-size`: Batch size (default: 1)
- `--gradient-accumulation`: Gradient accumulation steps (default: 2)

## Data Format
- Training data: Parquet files with tokenized sequences
- Mixture files: Text files with dataset names and proportions
- Expects data structure: `train_*.parquet`, `valid_*.parquet` files

## Memory Management
The framework includes extensive memory management:
- Gradient checkpointing support
- Optimizer state offloading to CPU
- Automatic garbage collection
- CUDA/MPS cache clearing
- Teacher model can be offloaded to separate GPU

## Distributed Training
Uses HuggingFace Accelerate for multi-GPU training:
```bash
accelerate launch --multi_gpu --gpu_ids all --num_processes 2 -m mldistill.standard [args]
```

## Development Environment Notes
- This project uses uv to manage dependencies. Prepend all commands that require the python environments with `uv run`.

## Claude Code Memories
- Don't try to make an update only add a newline at the end of a file. 