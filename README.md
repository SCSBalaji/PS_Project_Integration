# MobilePlantViT

A hybrid CNN-Transformer architecture for plant disease classification using the PlantVillage dataset.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PS_Project_Integration.git
cd PS_Project_Integration
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Check Your Setup

```bash
python utils/check_compute.py
```

### 5. Run Tests

```bash
pytest tests/ -v --ignore=tests/test_model.py
```

---

## Project Structure

```
PS_Project_Integration/
├── .github/workflows/    # CI/CD pipelines
├── blocks/               # Model building blocks
│   ├── ghost_conv.py     # Ghost Convolution
│   ├── fused_ir.py       # Fused Inverted Residual
│   ├── coord_att.py      # Coordinate Attention
│   ├── patch_embed.py    # Patch Embedding
│   ├── lda.py            # Linear Differential Attention
│   ├── bottleneck_ffn.py # Bottleneck FFN
│   ├── classifier.py     # Classification Head
│   └── model.py          # Full MobilePlantViT Model
├── config/
│   └── defaults.yaml     # Default configuration
├── tests/                # Unit and integration tests
├── utils/
│   ├── repro.py          # Reproducibility utilities
│   ├── logging_utils.py  # Experiment logging
│   ├── experiment.py     # Experiment management
│   └── check_compute.py  # Compute resource checker
├── experiments/          # Experiment outputs (gitignored)
├── CONTRIBUTING.md       # Contribution guidelines
├── COMPUTE_PLAN.md       # Compute resource planning
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Architecture Overview

MobilePlantViT combines efficient CNN blocks with transformer attention:

```
Input Image (224×224×3)
        ↓
   GhostConv (efficient feature extraction)
        ↓
   Fused Inverted Residual
        ↓
   Coordinate Attention
        ↓
   Patch Embedding
        ↓
   Positional Encoding
        ↓
   Linear Differential Attention
        ↓
   Residual LayerNorm
        ↓
   Bottleneck FFN
        ↓
   Global Average Pooling
        ↓
   Classifier Head
        ↓
   Output (38 classes)
```

---

## Configuration

All experiments use `config/defaults.yaml`:

```yaml
reproducibility:
  seed: 42
  cudnn_deterministic: true

dataset:
  name: "PlantVillage"
  num_classes: 38
  image_size: 224

training:
  num_epochs: 10
  learning_rate: 0.001
  batch_size: 64
```

---

## Running Experiments

### Create a New Experiment

```python
from utils import create_experiment, set_seed, load_config

# Create experiment with proper naming
exp = create_experiment(
    experiment_type="baseline",
    description="plantvillage_color"
)

# Access paths
print(exp.experiment_dir)
print(exp.get_checkpoint_path("best_model"))
```

### View TensorBoard Logs

```bash
tensorboard --logdir=experiments
```

---

## Testing

```bash
# Run all unit tests
pytest tests/ -v --ignore=tests/test_model.py

# Run with coverage
pytest tests/ -v --cov=blocks --cov=utils

# Run specific test
pytest tests/test_model_blocks.py::TestGhostConv -v
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Compute Resources

See [COMPUTE_PLAN.md](COMPUTE_PLAN.md) for resource planning and booking.

---

## License

[Add your license here]