# Contributing to MobilePlantViT

Thank you for your interest in contributing to MobilePlantViT! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Branch Naming Convention](#branch-naming-convention)
5. [Pull Request Guidelines](#pull-request-guidelines)
6. [Testing Requirements](#testing-requirements)
7. [Code Style Guidelines](#code-style-guidelines)
8. [Experiment Guidelines](#experiment-guidelines)
9. [Documentation Guidelines](#documentation-guidelines)

---

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. We are committed to providing a welcoming and inclusive environment.

---

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Git
- (Optional) CUDA-capable GPU

### Setup Development Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/PS_Project_Integration.git
cd PS_Project_Integration

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development tools
pip install pytest pytest-cov flake8 black

# 5. Verify setup
python run_smoke_test.py
pytest tests/ -v --ignore=tests/test_model.py
```

---

## Development Workflow

### 1. Create an Issue

Before starting work, create or find an issue describing the change:

- **Bug reports**: Describe the bug, steps to reproduce, expected behavior
- **Feature requests**: Describe the feature and its use case
- **Enhancements**: Describe the improvement and benefits

### 2. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Write code following our style guidelines
- Add or update tests
- Update documentation if needed

### 4. Test Your Changes

```bash
# Run all tests
pytest tests/ -v --ignore=tests/test_model.py

# Run specific tests
pytest tests/test_model_blocks.py -v

# Run smoke test
python run_smoke_test.py

# Check code style
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add: description of your change"
```

#### Commit Message Format

```
<type>: <short description>

[optional body]

[optional footer]
```

**Types:**
- `Add`: New feature
- `Fix`: Bug fix
- `Update`: Update existing feature
- `Remove`: Remove feature/code
- `Refactor`: Code refactoring
- `Test`: Add/update tests
- `Docs`: Documentation changes
- `CI`: CI/CD changes

**Examples:**
```
Add: GhostConv block with unit tests
Fix: Gradient flow issue in CoordAtt
Update: Increase default embed_dim to 256
Docs: Add architecture diagram to README
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Branch Naming Convention

| Branch Type | Pattern | Example |
|-------------|---------|---------|
| Feature | `feature/<name>` | `feature/ghost-conv` |
| Bug fix | `fix/<name>` | `fix/gradient-nan` |
| Experiment | `experiment/<name>` | `experiment/larger-embed-dim` |
| Documentation | `docs/<name>` | `docs/api-reference` |
| Refactor | `refactor/<name>` | `refactor/utils-cleanup` |

---

## Pull Request Guidelines

### PR Requirements

Every PR must include:

1. **Description**: Clear summary of changes
2. **Related Issue**: Link to the related issue (e.g., "Fixes #123")
3. **Tests**: Unit tests added or updated
4. **How to Test**: Instructions to verify the change
5. **Checklist**: Completed PR checklist

### PR Template

```markdown
## Description
[Brief description of changes]

## Related Issue
Fixes #[issue-number]

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Changes Made
- [Change 1]
- [Change 2]
- [Change 3]

## How to Test
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass locally
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings
- [ ] I have run the smoke test successfully

## Screenshots (if applicable)
[Add screenshots here]
```

### PR Review Process

1. At least **one approval** required before merging
2. All CI checks must pass
3. No unresolved review comments
4. Branch must be up-to-date with `main`

---

## Testing Requirements

### Required Tests for New Features

| Change Type | Required Tests |
|-------------|----------------|
| New block | Shape test, gradient test, forward test |
| Bug fix | Regression test proving fix |
| Model change | Integration test, smoke test |
| Utility function | Unit test with edge cases |

### Test Structure

```python
# tests/test_your_feature.py

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestYourFeature:
    """Tests for YourFeature."""
    
    def test_output_shape(self):
        """Test that output shape is correct."""
        from blocks import YourBlock
        
        block = YourBlock(inp=64, oup=128)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        
        assert out.shape == (2, 128, 32, 32)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        from blocks import YourBlock
        
        block = YourBlock(inp=64, oup=64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_model_blocks.py -v

# Specific test
pytest tests/test_model_blocks.py::TestGhostConv -v

# With coverage
pytest tests/ --cov=blocks --cov=utils --cov-report=html

# Smoke test only
pytest tests/test_smoke.py -v
```

---

## Code Style Guidelines

### General Rules

1. **Line length**: Maximum 120 characters
2. **Indentation**: 4 spaces (no tabs)
3. **Imports**: Organized in sections (stdlib, third-party, local)
4. **Docstrings**: Required for all public classes and functions
5. **Type hints**: Encouraged for function signatures

### Import Order

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local
from utils import set_seed, load_config
from blocks import GhostConv
```

### Docstring Format

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """
    Short description of function.
    
    Longer description if needed. Can span multiple lines
    and include more details about the function's behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: "default")
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = my_function(42, "test")
        >>> print(result)
        True
    """
    pass
```

### Class Docstring Format

```python
class MyBlock(nn.Module):
    """
    Short description of the block.
    
    Longer description of what this block does, its purpose
    in the architecture, and any important details.
    
    Args:
        inp: Number of input channels
        oup: Number of output channels
        stride: Convolution stride (default: 1)
        
    Attributes:
        conv: Main convolution layer
        bn: Batch normalization layer
        
    Example:
        >>> block = MyBlock(inp=64, oup=128)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([1, 128, 32, 32])
    """
    pass
```

### Linting

```bash
# Check for critical errors
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Check all style issues
flake8 . --count --max-complexity=10 --max-line-length=120 --statistics

# Auto-format with black (optional)
black .
```

---

## Experiment Guidelines

### Naming Convention

```
experiments/<YYYYMMDD>_<type>_<description>/
```

**Examples:**
```
experiments/20251129_baseline_plantvillage_color/
experiments/20251130_ablation_no_coordatt/
experiments/20251201_hyperparam_lr_sweep/
```

### Required Files

Every experiment must contain:

| File | Description | Required |
|------|-------------|----------|
| `config_used.yaml` | Exact configuration used | ✅ Yes |
| `checkpoints/best_model.pth` | Best model weights | ✅ Yes |
| `artifacts/training_history.json` | Training metrics | ✅ Yes |
| `artifacts/final_summary.json` | Final results | ✅ Yes |
| `README.md` | Notes about the run | Optional |

### Using ExperimentManager

```python
from utils import create_experiment, load_config

# Create experiment
exp = create_experiment(
    experiment_type="baseline",
    description="plantvillage_color"
)

# Access paths
checkpoint_path = exp.get_checkpoint_path("best_model")
tensorboard_dir = exp.get_tensorboard_dir()

# Save artifacts
exp.save_artifact("metrics", {"accuracy": 0.95})

# Finalize
exp.finalize({"best_accuracy": 0.95, "total_epochs": 10})
```

---

## Documentation Guidelines

### When to Update Documentation

- Adding new features
- Changing existing behavior
- Adding new configuration options
- Fixing bugs that change expected behavior

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start |
| `CONTRIBUTING.md` | Contribution guidelines |
| `COMPUTE_PLAN.md` | Compute resources |
| `CODE_OF_CONDUCT.md` | Community guidelines |

### Code Documentation

- Add docstrings to all public classes and functions
- Include usage examples in docstrings
- Update `__all__` in `__init__.py` when adding exports

---

## Questions?

If you have questions:

1. Check existing [issues](https://github.com/YOUR_USERNAME/PS_Project_Integration/issues)
2. Search closed issues and PRs
3. Open a new issue with the `question` label

Thank you for contributing! 🎉