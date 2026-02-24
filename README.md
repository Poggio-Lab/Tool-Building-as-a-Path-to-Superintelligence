# GF2 Code

This repository contains code for generating boolean circuit datasets, simulating Bayesian estimators, running LLM inference, and plotting results.

## Setup

Ensure you have Python installed. It is recommended to use `uv` for dependency management and running scripts, as suggested in the file headers.

### Dependencies

Run `uv pip install -r requirements.txt` to install dependencies.

## Usage

### 1. Generate Datasets

Generate the experimental datasets using `generate_dataset.py`.

```bash
# Generate all 4 datasets (full/llm × diagnostic/adversarial)
uv run generate_dataset.py --generate-both
```

This will create JSON files in the `storage/` directory.

### 2. Run Estimator Simulations

Run the Bayesian estimators (A, B, C, D) on the generated datasets using `simulate.py`.

```bash
# Run on all full datasets
uv run simulate.py
```

This saves the results (gamma values) to `storage/`.

### 3. Run LLM Inference (Optional)

To run LLM inference using vLLM, use `vllm_deeptest.py`.

```bash
# Example: Run on LLM adversarial dataset
uv run vllm_deeptest.py --input experiments_llm_adversarial.json --model Qwen/Qwen3-4B-Instruct-2507
```

### 4. Plot Results

Generate plots from the simulation and LLM results using `plot_all.py`.

```bash
uv run plot_all.py
```

Plots will be saved to the `plots/` directory.

## File Structure

- `generate_dataset.py`: Generates boolean circuit experiments.
- `simulate.py`: Runs Bayesian estimators on the datasets.
- `vllm_deeptest.py`: Runs local vLLM inference.
- `plot_all.py`: Generates all figures for the paper.
- `plot_utils.py`: Shared utilities and configuration for plotting.
- `storage/`: Directory for datasets and simulation results.
- `plots/`: Directory for generated plots.
- `llm_results/`: Directory for LLM inference results.
