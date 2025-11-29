# Contributing to MobilePlantViT

This document outlines the conventions and guidelines for contributing to this project.

---

## Table of Contents

1. [Branch Workflow](#branch-workflow)
2. [Pull Request Guidelines](#pull-request-guidelines)
3. [Experiment Naming Convention](#experiment-naming-convention)
4. [Artifact Structure](#artifact-structure)
5. [Running Tests](#running-tests)
6. [Code Style](#code-style)

---

## Branch Workflow

1. **Main branch**: `main` is protected. Never push directly.
2. **Feature branches**: Create from `main` with naming:
   - `feature/<feature-name>` for new features
   - `fix/<bug-name>` for bug fixes
   - `experiment/<experiment-name>` for experimental changes

3. **Branch protection rules**:
   - Require pull request reviews before merging
   - Require CI checks to pass

---

## Pull Request Guidelines

Every PR must include:

1. **Description**: Clear summary of changes
2. **Linked Issue**: Reference the related issue (e.g., "Fixes #123")
3. **Tests**: Unit tests added or updated
4. **How to Run**: Brief instructions to test the changes

### PR Template

```markdown
## Description
[Brief description of changes]

## Related Issue
Fixes #[issue-number]

## Changes Made
- [Change 1]
- [Change 2]

## How to Test
1. [Step 1]
2. [Step 2]

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
```

---

## Experiment Naming Convention

All experiments must follow this naming scheme:

```
experiments/<YYYYMMDD>_<experiment-type>_<description>/
```

### Examples

```
experiments/20251129_baseline_plantvillage_color/
experiments/20251130_ablation_no_coordatt/
experiments/20251201_hyperparam_lr_sweep/
```

### Experiment Types

| Type | Description |
|------|-------------|
| `baseline` | Standard training run |
| `ablation` | Removing/modifying components |
| `hyperparam` | Hyperparameter tuning |
| `debug` | Debug/test runs |
| `final` | Final production runs |

---

## Artifact Structure

Every experiment folder must contain:

```
experiments/<experiment-name>/
├── config_used.yaml          # REQUIRED: Exact config used for this run
├── checkpoints/
│   ├── best_model.pth        # Best model checkpoint
│   └── latest_checkpoint.pth # Latest checkpoint
├── logs/
│   └── tensorboard/          # TensorBoard logs
├── artifacts/
│   ├── training_history.json # Loss/accuracy per epoch
│   ├── training_curves.png   # Loss/accuracy plots
│   ├── confusion_matrix.png  # Final confusion matrix
│   └── final_summary.json    # Final metrics summary
└── README.md                 # Optional: Notes about this run
```

### Required Files

| File | Description | Required |
|------|-------------|----------|
| `config_used.yaml` | Exact configuration snapshot | ✅ Yes |
| `best_model.pth` | Best model weights | ✅ Yes |
| `training_history.json` | Training metrics per epoch | ✅ Yes |
| `final_summary.json` | Final results summary | ✅ Yes |

### config_used.yaml Must Include

```yaml
# Reproducibility (REQUIRED)
reproducibility:
  seed: 42
  cudnn_deterministic: true
  cudnn_benchmark: false

# Dataset info
dataset:
  name: "PlantVillage"
  variant: "color"
  num_classes: 38

# Training params
training:
  num_epochs: 10
  learning_rate: 0.001
  batch_size: 64

# Runtime metadata (auto-generated)
_runtime:
  experiment_dir: "experiments/..."
  start_time: "2025-11-29T10:30:00"
  torch_version: "2.0.0"
  cuda_available: true
```

---

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Only Unit Tests (faster)

```bash
pytest tests/ -v --ignore=tests/test_model.py
```

### Run Specific Test File

```bash
pytest tests/test_model_blocks.py -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=blocks --cov=utils --cov-report=html
```

---

## Code Style

### General Guidelines

1. **Line length**: Maximum 120 characters
2. **Docstrings**: Required for all public classes and functions
3. **Type hints**: Encouraged for function signatures
4. **Imports**: Organized (stdlib, third-party, local)

### Tools

- **Linting**: `flake8`
- **Formatting**: `black` (optional auto-format)

### Run Linting

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Auto-format (optional)

```bash
black .
```

---

## Quick Start for Contributors

1. Clone and create branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests to verify setup:
   ```bash
   pytest tests/ -v --ignore=tests/test_model.py
   ```

4. Make changes and add tests

5. Commit and push:
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature
   ```

6. Create PR on GitHub

---

## Questions?

If you have questions, open an issue or contact the team lead.