# Overview: Generating Research Paper Visualizations for MobilePlantViT

Based on the reference tables from the research paper, here's a theoretical overview of what you need to do to generate similar publication-ready visualizations.

---

## 📊 Types of Tables/Visualizations Needed

Looking at your reference images, there are **4 main types** of result tables you need:

### 1. **Performance Summary Table (TABLE II style)**
Shows your model's performance across multiple metrics on the PlantVillage dataset.

| What to include |
|-----------------|
| Dataset name, Test size |
| Accuracy (%) |
| Precision (Macro & Weighted) |
| Recall (Macro & Weighted) |
| F1-score (Macro & Weighted) |
| AUC (OvR - One vs Rest) |

---

### 2. **Transfer Learning / Pre-training Impact Table (TABLE III style)**
Shows improvement when using pre-trained weights vs training from scratch.

| What to compare |
|-----------------|
| Baseline (random init) vs Pre-trained |
| Show improvement with ↑ arrows |
| All metrics: Accuracy, Precision, Recall, F1 |

---

### 3. **Ablation Study Table (TABLE IV style)**
Shows contribution of each architectural component (your upgraded blocks).

| Components to ablate |
|---------------------|
| With/Without GhostConv |
| With/Without CoordAtt |
| With/Without Linear Differential Attention |
| With/Without Bottleneck FFN |
| Group convolution configurations (1-1-1 vs 1-2-4) |

---

### 4. **Model Comparison Table (TABLE V style)**
Compares your MobilePlantViT against other lightweight ViTs.

| Models to compare |
|-------------------|
| MobilePlantViT (yours) |
| MobileNetV2 (baseline) |
| MobileViT variants |
| EfficientNet-B0 |
| Other lightweight models |

---

## 🔄 What You Need to Do (High-Level Steps)

```markdown
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH VISUALIZATION FLOW               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Collect Metrics                                     │
│  ─────────────────────                                       │
│  • Run evaluation on test set                                │
│  • Calculate ALL metrics (not just accuracy)                 │
│  • Store per-class and aggregate results                     │
│                                                              │
│  Step 2: Run Ablation Experiments                            │
│  ───────────────────────────────                             │
│  • Train model variants with components removed              │
│  • Compare: Full model vs each ablated version               │
│  • Quantify contribution of each block                       │
│                                                              │
│  Step 3: Train Comparison Models                             │
│  ─────────────────────────────                               │
│  • Train MobileNetV2 baseline (same hyperparameters)         │
│  • Optionally train other lightweight models                 │
│  • Ensure fair comparison (same data, augmentation)          │
│                                                              │
│  Step 4: Generate LaTeX Tables                               │
│  ────────────────────────────                                │
│  • Format results in publication-ready LaTeX                 │
│  • Add highlighting for best results                         │
│  • Include statistical significance if applicable            │
│                                                              │
│  Step 5: Create Figures                                      │
│  ─────────────────────                                       │
│  • Training curves comparison                                │
│  • Parameter efficiency plots                                │
│  • Confusion matrix (publication quality)                    │
│  • Architecture diagram                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```



---

## 📋 Metrics You Need to Calculate

From your current training, you have:
- ✅ Accuracy
- ✅ Per-class accuracy
- ✅ Confusion matrix

**You still need to calculate:**
- ⬜ Macro Precision
- ⬜ Weighted Precision  
- ⬜ Macro Recall
- ⬜ Weighted Recall
- ⬜ Macro F1-score
- ⬜ Weighted F1-score
- ⬜ AUC-ROC (One vs Rest)

---

## 🔬 Ablation Experiments Required

To create TABLE IV (ablation study), you need to train **multiple model variants**:

| Experiment | What to remove/modify |
|------------|----------------------|
| Exp 1 | Remove CoordAtt (use standard attention) |
| Exp 2 | Remove GhostConv (use standard conv) |
| Exp 3 | Remove Linear Differential Attention (use standard MHSA) |
| Exp 4 | Different group conv configs |
| Exp 5 | Full model (baseline) |

---

## 📈 Additional Visualizations for Paper

Beyond tables, research papers typically include:

1. **Training Curves** - Loss & accuracy over epochs (you have this)
2. **Architecture Diagram** - Visual representation of MobilePlantViT
3. **Attention Visualization** - Where the model focuses on leaf images
4. **t-SNE/UMAP** - Feature space visualization
5. **Grad-CAM** - Class activation maps
6. **Parameter vs Accuracy Plot** - Efficiency comparison

---

## 🎯 Summary: Your Action Plan

| Phase | Tasks | Estimated Effort |
|-------|-------|------------------|
| **Phase 1** | Calculate all metrics from current model | 1-2 hours |
| **Phase 2** | Run ablation experiments (4-5 variants) | 1-2 days (training time) |
| **Phase 3** | Train comparison models | 1 day |
| **Phase 4** | Generate LaTeX tables & figures | 2-3 hours |
| **Phase 5** | Create additional visualizations | 2-3 hours |

---

**Ready for the detailed implementation steps?** Let me know which part you'd like to start with:

1. **Calculating all metrics** from your existing trained model
2. **Setting up ablation experiments**
3. **Creating LaTeX table generators**
4. **Generating publication-ready figures**