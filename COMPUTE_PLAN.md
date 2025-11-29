# Compute Plan for MobilePlantViT

This document outlines the compute resources available for training and experimentation.

---

## Table of Contents

1. [Available Resources](#available-resources)
2. [Resource Specifications](#resource-specifications)
3. [Training Time Estimates](#training-time-estimates)
4. [Resource Booking](#resource-booking)
5. [Best Practices](#best-practices)

---

## Available Resources

### Local Machine(s)

| Machine | Owner | GPU | VRAM | Status |
|---------|-------|-----|------|--------|
| Local Workstation | [Your Name] | [Your GPU, e.g., RTX 3060] | [VRAM, e.g., 12GB] | Available |

### Cloud Resources (if applicable)

| Platform | Account | GPU Type | Hours Available | Cost |
|----------|---------|----------|-----------------|------|
| Google Colab | Free | T4 | ~12h/day | Free |
| Google Colab Pro | [If purchased] | T4/V100 | Priority | $10/month |
| Kaggle | Free | P100/T4 | 30h/week | Free |
| AWS/GCP | [If available] | [Instance type] | [Budget] | [Cost] |

### University/Lab Resources (if applicable)

| Resource | Access | GPU Type | Booking Required |
|----------|--------|----------|------------------|
| Lab Server | [Access method] | [GPU type] | Yes/No |
| HPC Cluster | [Access method] | [GPU type] | Yes/No |

---

## Resource Specifications

### Minimum Requirements

- **GPU**: CUDA-capable GPU with 6GB+ VRAM
- **RAM**: 16GB system RAM
- **Storage**: 10GB free disk space
- **Python**: 3.10 or 3.11

### Recommended Specifications

- **GPU**: NVIDIA RTX 3060 or better (12GB+ VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 50GB SSD
- **Python**: 3.11

### Check Your GPU

Run this command to check your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"
```

---

## Training Time Estimates

### PlantVillage Dataset (38 classes, ~54,000 images)

| Configuration | Batch Size | Time per Epoch | Total (10 epochs) |
|---------------|------------|----------------|-------------------|
| RTX 3060 (12GB) | 64 | ~3-5 min | ~30-50 min |
| RTX 3080 (10GB) | 64 | ~2-3 min | ~20-30 min |
| T4 (Colab) | 32 | ~5-8 min | ~50-80 min |
| V100 (Colab Pro) | 64 | ~2-3 min | ~20-30 min |
| CPU only | 16 | ~30-60 min | ~5-10 hours |

### Memory Usage Estimates

| Batch Size | Approx. VRAM Usage |
|------------|-------------------|
| 16 | ~4 GB |
| 32 | ~6 GB |
| 64 | ~10 GB |
| 128 | ~18 GB |

---

## Resource Booking

### Booking Calendar

Use this section to coordinate GPU access among team members.

| Date | Time Slot | Resource | User | Experiment |
|------|-----------|----------|------|------------|
| YYYY-MM-DD | HH:MM-HH:MM | [Resource] | [Name] | [Experiment name] |
| | | | | |

### Booking Rules

1. **Book in advance**: Reserve GPU time at least 24 hours ahead for long runs
2. **Release when done**: Free up resources if finishing early
3. **Tag experiments**: Use `debug` type for short test runs
4. **Communicate**: Notify team on Slack/Discord when starting long runs

---

## Best Practices

### Before Training

1. **Test locally first**: Run a quick debug experiment (1-2 epochs) on CPU or small GPU
2. **Check disk space**: Ensure enough space for checkpoints and logs
3. **Verify data**: Confirm dataset is downloaded and accessible
4. **Set seeds**: Always use config seeds for reproducibility

### During Training

1. **Monitor GPU usage**: Use `nvidia-smi` or `watch -n 1 nvidia-smi`
2. **Log to TensorBoard**: Track metrics in real-time
3. **Save checkpoints**: Enable automatic checkpoint saving
4. **Handle interruptions**: Use latest checkpoint to resume if needed

### After Training

1. **Save artifacts**: Ensure all artifacts are saved properly
2. **Log results**: Update experiment tracking (W&B/TensorBoard)
3. **Clean up**: Remove unnecessary checkpoints to save space
4. **Document**: Add notes to experiment README

### Quick Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU continuously
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check Python/PyTorch setup
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Colab/Kaggle Quick Start

### Google Colab

```python
# Mount Google Drive (for data persistence)
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/YOUR_USERNAME/PS_Project_Integration.git
%cd PS_Project_Integration

# Install dependencies
!pip install -r requirements.txt

# Check GPU
!nvidia-smi
```

### Kaggle

```python
# Enable GPU in Settings > Accelerator > GPU

# Install additional dependencies
!pip install wandb tensorboard

# Check GPU
!nvidia-smi
```

---

## Emergency Contacts

| Role | Name | Contact |
|------|------|---------|
| Team Lead | [Name] | [Email/Slack] |
| GPU Admin | [Name] | [Email/Slack] |

---

## Notes

- Update this document as resources change
- Add new team members to booking calendar
- Document any issues with specific resources