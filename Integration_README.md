# MobilePlantViT Source Code - Kaggle Dataset

## Dataset Contents

This dataset contains the complete source code for the MobilePlantViT model,
a lightweight hybrid CNN-Transformer architecture for plant disease classification.

### Directory Structure
```
src/
├── __init__.py                 # Package initialization
├── blocks/                     # Building block modules
│   ├── __init__.py
│   ├── attention.py            # Linear Differential Attention
│   ├── classifier.py           # GAP and Classifier Head
│   ├── coord_attention.py      # Coordinate Attention
│   ├── ffn.py                  # Bottleneck FFN
│   ├── fused_ir.py             # Fused Inverted Residual
│   ├── ghost_conv.py           # Ghost Convolution
│   ├── patch_embed.py          # Patch Embedding
│   ├── positional_encoding.py  # Positional Encoding
│   └── utils.py                # Utility functions
└── models/
    ├── __init__.py
    └── mobile_plant_vit.py     # Main model implementation
```

## How to Use in Kaggle Notebook

### Step 1: Add Dataset to Notebook

1. Open your Kaggle notebook
2. Click **"Add Data"** button (right panel)
3. Go to **"Your Datasets"** tab
4. Search for **"mobileplant-vit-source"**
5. Click **"Add"** to add the dataset

### Step 2: Import in Your Notebook

Add this code to your first cell:

```python
import sys
import os

# Add MobilePlantViT source to path
SOURCE_DATASET = 'mobileplant-vit-source'
source_path = f'/kaggle/input/{SOURCE_DATASET}'

if os.path.exists(source_path):
    sys.path.insert(0, source_path)
    print(f"✅ Source added: {source_path}")
else:
    raise FileNotFoundError(f"Add '{SOURCE_DATASET}' dataset to this notebook")

# Import MobilePlantViT
from src.models import (
    MobilePlantViT,
    MobilePlantViTConfig,
    mobileplant_vit_tiny,
    mobileplant_vit_small,
    mobileplant_vit_base,
    mobileplant_vit_large,
)
print("✅ MobilePlantViT imported!")
```

### Step 3: Create Model

```python
# Using factory function (recommended)
model = mobileplant_vit_base(num_classes=38)

# Or with custom configuration
config = MobilePlantViTConfig(
    num_classes=38,
    embed_dim=256,
    num_heads=8,
)
model = MobilePlantViT(config)
```

## Model Variants

| Variant | Parameters | embed_dim | num_heads | Use Case |
|---------|------------|-----------|-----------|----------|
| tiny    | ~250K      | 128       | 4         | Edge devices |
| small   | ~500K      | 192       | 6         | Mobile deployment |
| base    | ~867K      | 256       | 8         | Default (recommended) |
| large   | ~2M        | 384       | 12        | Higher capacity |

## Requirements

- PyTorch >= 1.9.0
- No additional dependencies beyond standard Kaggle environment

## Troubleshooting

**Import Error:**
- Verify dataset is added to notebook
- Check that `sys.path.insert(0, source_path)` is called before imports

**Model Error:**
- Ensure `num_classes` matches your dataset
- For PlantVillage: `num_classes=38`

## License

Apache 2.0