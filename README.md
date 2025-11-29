# MobilePlantViT 🌿

A hybrid CNN-Transformer architecture for efficient plant disease classification using the PlantVillage dataset.

[![CI Pipeline](https://github.com/YOUR_USERNAME/PS_Project_Integration/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/PS_Project_Integration/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Running Experiments](#running-experiments)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

MobilePlantViT is a lightweight, efficient deep learning model designed for plant disease classification. It combines the efficiency of MobileNet-style convolutions with the global attention capabilities of Vision Transformers.

### Key Features

- **Hybrid Architecture**: Combines CNN and Transformer blocks
- **Efficient Design**: Uses Ghost Convolutions and Fused Inverted Residuals
- **Attention Mechanisms**: Coordinate Attention + Linear Differential Attention
- **Reproducible**: Full seed control and experiment tracking
- **Well-Tested**: Comprehensive unit and smoke tests

### Target Dataset

- **PlantVillage Dataset**: 38 classes, ~54,000 images
- **Input Size**: 224×224 RGB images
- **Task**: Multi-class plant disease classification

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MobilePlantViT Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Image (224×224×3)                                       │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                           │
│   │   GhostConv     │  Efficient feature extraction             │
│   │   (3 → 64)      │  with ghost modules                       │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Fused Inverted │  Mobile-style inverted residual           │
│   │    Residual     │  with fused operations                    │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Coordinate     │  Spatial attention mechanism              │
│   │   Attention     │  (horizontal + vertical)                  │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Patch Embedding │  Convert features to sequence             │
│   │  + Pos. Enc.    │  with learnable positions                 │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │    Linear       │  Efficient attention with                 │
│   │  Differential   │  differential mechanism                   │
│   │   Attention     │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Residual + LN   │  Skip connection with                     │
│   │                 │  Layer Normalization                      │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Bottleneck FFN │  Feed-forward with                        │
│   │                 │  bottleneck design                        │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Global Average │  Sequence to vector                       │
│   │    Pooling      │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  Classifier     │  FC → Softmax                             │
│   │    Head         │  (256 → 38 classes)                       │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   Output: Class Probabilities (38)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Building Blocks

| Block | Description | Key Features |
|-------|-------------|--------------|
| **GhostConv** | Ghost Convolution Module | Generates features with fewer parameters |
| **Fused-IR** | Fused Inverted Residual | Efficient mobile block with SE attention |
| **CoordAtt** | Coordinate Attention | Captures spatial dependencies |
| **PatchEmbed** | Patch Embedding | Converts CNN features to sequence |
| **LDA** | Linear Differential Attention | Efficient self-attention variant |
| **BottleneckFFN** | Bottleneck Feed-Forward | Parameter-efficient FFN |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PS_Project_Integration.git
cd PS_Project_Integration
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check compute resources
python utils/check_compute.py

# Run smoke test
python run_smoke_test.py

# Run unit tests
pytest tests/ -v --ignore=tests/test_model.py
```

---

## 📁 Project Structure

```
PS_Project_Integration/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
│
├── blocks/                     # Model building blocks
│   ├── __init__.py
│   ├── ghost_conv.py           # Ghost Convolution
│   ├── fused_ir.py             # Fused Inverted Residual
│   ├── coord_att.py            # Coordinate Attention
│   ├── patch_embed.py          # Patch Embedding + Positional Encoding
│   ├── lda.py                  # Linear Differential Attention
│   ├── res_norm.py             # Residual LayerNorm Block
│   ├── bottleneck_ffn.py       # Bottleneck FFN
│   ├── classifier.py           # Classifier Head + GAP
│   ├── preprocessing.py        # Data preprocessing utilities
│   └── model.py                # Full MobilePlantViT Model
│
├── config/
│   └── defaults.yaml           # Default configuration
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_model_blocks.py    # Block unit tests
│   ├── test_model.py           # Model integration tests
│   ├── test_reproducibility.py # Reproducibility tests
│   └── test_smoke.py           # End-to-end smoke tests
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── repro.py                # Reproducibility utilities
│   ├── logging_utils.py        # Experiment logging
│   ├── experiment.py           # Experiment management
│   ├── init_run.py             # Run initialization helper
│   └── check_compute.py        # Compute resource checker
│
├── experiments/                # Experiment outputs (gitignored)
│   └── YYYYMMDD_type_name/
│       ├── config_used.yaml
│       ├── checkpoints/
│       ├── logs/
│       └── artifacts/
│
├── .flake8                     # Flake8 configuration
├── .gitignore                  # Git ignore rules
├── CONTRIBUTING.md             # Contribution guidelines
├── COMPUTE_PLAN.md             # Compute resource planning
├── CODE_OF_CONDUCT.md          # Code of conduct
├── LICENSE                     # License file
├── pyproject.toml              # Project configuration
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── run_smoke_test.py           # Standalone smoke test
└── README.md                   # This file
```

---

## 💻 Installation

### Prerequisites

- Python 3.10 or 3.11 (recommended)
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Detailed Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/PS_Project_Integration.git
cd PS_Project_Integration

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 6. Check project setup
python utils/check_compute.py
```

### GPU Setup (Optional)

If you have an NVIDIA GPU:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA (adjust version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧪 Running Tests

### Run All Unit Tests

```bash
pytest tests/ -v --ignore=tests/test_model.py
```

### Run Specific Test Files

```bash
# Block tests only
pytest tests/test_model_blocks.py -v

# Reproducibility tests
pytest tests/test_reproducibility.py -v

# Smoke tests
pytest tests/test_smoke.py -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=blocks --cov=utils --cov-report=html
# Open htmlcov/index.html in browser
```

### Run Standalone Smoke Test

```bash
python run_smoke_test.py
```

---

## 🔬 Running Experiments

### Initialize a New Experiment

```python
from utils import init_training_run

# Initialize with proper reproducibility
config, exp_dir, logger = init_training_run(
    experiment_name="my_experiment",
    experiment_type="baseline",
    use_tensorboard=True
)

# Access paths
print(f"Experiment directory: {exp_dir}")
print(f"Config: {config}")
```

### Experiment Types

| Type | Use Case |
|------|----------|
| `baseline` | Standard training runs |
| `ablation` | Removing/modifying components |
| `hyperparam` | Hyperparameter tuning |
| `debug` | Quick debug/test runs |
| `final` | Final production runs |

### View TensorBoard Logs

```bash
tensorboard --logdir=experiments
# Open http://localhost:6006 in browser
```

---

## ⚙️ Configuration

### Default Configuration (`config/defaults.yaml`)

```yaml
# Reproducibility
reproducibility:
  seed: 42
  cudnn_deterministic: true
  cudnn_benchmark: false

# Dataset
dataset:
  name: "PlantVillage"
  num_classes: 38
  image_size: 224
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Model
model:
  name: "MobilePlantViT"
  embed_dim: 256
  num_heads: 8
  patch_size: 14

# Training
training:
  num_epochs: 10
  learning_rate: 0.001
  batch_size: 64
  optimizer: "adamw"
  weight_decay: 0.01
```

### Override Configuration

```python
from utils import load_config

# Load and modify
config = load_config("config/defaults.yaml")
config['training']['learning_rate'] = 0.0005
config['training']['batch_size'] = 32
```

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a PR.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/ -v`
5. Commit: `git commit -m "Add your feature"`
6. Push: `git push origin feature/your-feature`
7. Create a Pull Request

### Requirements for PRs

- All tests must pass
- Code must follow style guidelines
- Include tests for new features
- Update documentation as needed

---

## 📚 Documentation

- [Contributing Guidelines](CONTRIBUTING.md)
- [Compute Plan](COMPUTE_PLAN.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PlantVillage dataset creators
- PyTorch team
- MobileNet and Vision Transformer paper authors

---

## 📧 Contact

For questions or issues, please [open an issue](https://github.com/YOUR_USERNAME/PS_Project_Integration/issues) on GitHub.