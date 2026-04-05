# Bandwidth-Efficient Multi-Agent Communication Implementation

Implementation of:
**Farooq, A. & Iqbal, K. (2026). "Bandwidth-Efficient Multi-Agent Communication through Information Bottleneck and Vector Quantization." arXiv:2602.02035v1**

## Paper Summary

This implementation reproduces the GVQ (Gated Vector Quantization) method that achieves:
- **38.8% success rate** (181.8% improvement over no-communication baseline)
- **800 bits/episode bandwidth** (41.4% reduction vs full communication)
- **Pareto AUC of 0.198** (dominance across success-bandwidth spectrum)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Running the Code

### Quick Start (Single Seed)
```bash
python main.py
```

### Full Experiment (8 Seeds as in Paper)

The default `main.py` runs all 8 seeds automatically. To modify:
```python
# In config.py, change:
NUM_SEEDS = 8  # Set to 1 for quick testing
NUM_EPISODES = 2000  # Reduce for faster testing
```

### Training Output

Results are saved to:
- `results/results_seed_X.json` - Individual seed results
- `results/aggregate_results.json` - Mean ± std across seeds
- `runs/gvq_seed_X/` - TensorBoard logs
- `checkpoints/best_model_seed_X.pt` - Best model checkpoints

### Monitoring Training
```bash
# Launch TensorBoard
tensorboard --logdir=runs

# Open browser to http://localhost:6006
```

## Project Structure
