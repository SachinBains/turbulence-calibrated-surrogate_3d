# Turbulence-Calibrated-Surrogate

A 3D turbulence surrogate model with uncertainty quantification and interpretability analysis.

## Quick Start

See `configs/` for experiment configurations and `scripts/` for CLI entrypoints.

## Interpretability Suite

### Local Attributions
Generate per-sample saliency maps using Integrated Gradients, GradientSHAP, or 3D occlusion:

```bash
# Integrated Gradients for baseline model
python -m scripts.explain_local --config configs/E1_hit_baseline.yaml --split val --method ig --n 2

# 3D Occlusion for MC dropout model  
python -m scripts.explain_local --config configs/E4_hit_ab_dropout.yaml --split test --method occlusion --n 2

# GradientSHAP
python -m scripts.explain_local --config configs/E2_hit_bayes.yaml --split val --method gradshap --n 4
```

### Faithfulness Evaluation
Measure attribution quality via top-k ablation curves:

```bash
# Test faithfulness of Integrated Gradients
python -m scripts.faithfulness --config configs/E1_hit_baseline.yaml --split val --method ig --k_list 0.05 0.1 0.2

# Test occlusion faithfulness
python -m scripts.faithfulness --config configs/E4_hit_ab_dropout.yaml --split test --method occlusion --k_list 0.1 0.2 0.3
```

### Global Feature Importance
Analyze which turbulence features correlate with model error or uncertainty:

```bash
# Feature importance for prediction error (baseline models)
python -m scripts.explain_global --config configs/E1_hit_baseline.yaml --split test --target error

# Feature importance for uncertainty (MC models)
python -m scripts.explain_global --config configs/E4_hit_ab_dropout.yaml --split test --target sigma
```

## Device Usage

All scripts default to CPU. Add `--cuda` flag to use GPU:

```bash
# Run on CPU (default)
python -m scripts.run_eval --config configs/E1_hit_baseline.yaml

# Run on GPU
python -m scripts.run_eval --config configs/E1_hit_baseline.yaml --cuda
```
