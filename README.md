# Adaptive Rank LoRA: Fine-Tuning Large Language Models using LoRA with Adaptive Rank Allocation Based on Spectral Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel Parameter Efficient Fine-Tuning (PEFT) method that dynamically assigns optimal LoRA ranks to neural network layers using Heavy-Tailed Self-Regularization (HT-SR) theory and empirical spectral density analysis.

## 🎯 Overview

Fine-tuning large language models (LLMs) efficiently is a crucial challenge due to their immense size and computational demands. While LoRA (Low-Rank Adaptation) offers a parameter-efficient alternative, determining the optimal rank per layer remains an open problem. This work proposes a theoretically grounded approach to assign LoRA ranks based on the spectral properties of weight matrices.

### Key Innovation

Traditional LoRA implementations assign fixed or heuristically determined ranks to all adapted layers. Our method leverages **Heavy-Tailed Self-Regularization (HT-SR) theory** to:
- 🔬 **Analyze** empirical spectral density (ESD) of weight matrices
- 📊 **Quantify** layer importance using power-law exponents and eigenvalue outliers
- ⚖️ **Assign** ranks dynamically based on mathematical principles
- 🚀 **Achieve** superior performance with fewer trainable parameters

## 🔬 Methodology

Our approach is inspired by [AlphaPruning](https://github.com/haiquanlu/AlphaPruning) and leverages insights from Heavy-Tailed Self-Regularization (HT-SR) theory to dynamically assign LoRA ranks based on spectral properties of weight matrices.

### Spectral Analysis Pipeline

1. **Weight Matrix Correlation Analysis**
   - For each layer's weight matrix W_l ∈ R^(M×N), compute correlation matrix X_l = W_l^T W_l
   - Extract eigenvalues to characterize the spectral properties

2. **Power-Law Alpha Estimation (α_hill)**
   - Use Hill estimator with median method to estimate power-law exponent
   - α_hill = 1 + k / Σ(ln(λ_i / λ_threshold)) where k = ⌈n/2⌉
   - Lower α values indicate more heavy-tailed distributions (well-structured layers)
   - Higher α values suggest layers needing more adaptation

3. **Marchenko-Pastur Spike Detection**
   - Calculate theoretical bulk edge: λ_max^MP = σ_MP² (1 + 1/√Q)²
   - Count eigenvalues exceeding this threshold (spectral spikes)
   - Fewer spikes indicate layers with less learned structure

### Adaptive Rank Assignment Strategy

**Step 1: Metric Normalization**
```
α_hill_norm = (α_hill - min(α_hill)) / (max(α_hill) - min(α_hill) + ε)
N_s_norm = 1 - (N_s - min(N_s)) / (max(N_s) - min(N_s) + ε)
```

**Step 2: Composite Scoring**
```
Rank_Adoption_Score = w_hill × α_hill_norm + w_spikes × N_s_norm
```
Default weights: w_hill = 0.7, w_spikes = 0.3

**Step 3: Linear Rank Assignment**
```
R_l = R_min + (R_max - R_min) × (RA_l - RA_min) / (RA_max - RA_min + ε)
```

### Key Insights

- **Layers with higher α_hill**: Less structured, require more adaptation → Higher LoRA ranks
- **Layers with fewer spikes**: Less significant learned features → Higher LoRA ranks  
- **Dynamic allocation**: Computational resources focus on layers that benefit most from adaptation

## 🚀 Features

- ✅ **Automatic Rank Assignment**: No manual hyperparameter tuning
- ✅ **Layer Selection**: Identifies optimal layers for adaptation  
- ✅ **Multiple LoRA Variants**: Supports QLoRA, RSLoRA, DoRA
- ✅ **Comprehensive Monitoring**: Built-in system and performance tracking
- ✅ **Flexible Configuration**: Extensive customization options
- ✅ **Memory Efficient**: Reduced parameter count with better performance

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 16GB+ GPU memory



### Quick Installation

```bash
# Clone the repository
git clone https://github.com/UdiBhaskar/Adaptive-Rank-LoRA.git
cd Adaptive-Rank-LoRA

# Install all dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
# Test basic imports
import torch
import transformers
import peft
from src.adaptive_rank_assignment import get_adaptive_lora_config

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")
print("✅ Installation successful!")
```

## 🎮 Quick Start

### Basic Usage

```python
from src.adaptive_rank_assignment import get_adaptive_lora_config
from transformers import AutoModelForCausalLM
from peft import LoraConfig

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model-name")

# Get adaptive LoRA configuration
rank_pattern, alpha_pattern, target_regex = get_adaptive_lora_config(
    model,
    layer_selection_percentile=0.5,  # Adapt top 50% of layers
    minimum_rank=4,
    maximum_rank=64,
    rank_scaling_method="linear"
)

# Create PEFT config with adaptive ranks
peft_config = LoraConfig(
    target_modules=target_regex,
    rank_pattern=rank_pattern,
    alpha_pattern=alpha_pattern,
    task_type="CAUSAL_LM"
)
```

### Training Script

```bash
python src/lora_dynamic_rank.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --train_dataset_path "data/math10k_train.parquet" \
    --val_dataset_path "data/math10k_val.parquet" \
    --output_dir "outputs/adaptive_lora" \
    --run_name "llama7b_math_adaptive" \
    --top_n_percentile 0.5 \
    --min_rank 4 \
    --max_rank 64 \
    --rank_scaling "linear" \
    --alpha_factor 2 \
    --w_hill 0.7 \
    --w_mp_spikes 0.3 \
    --use_peft \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --warmup_steps 100
```

## ⚙️ Configuration

### Rank Assignment Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `top_n_percentile` | Fraction of layers to adapt | 0.5 | 0.1-1.0 |
| `min_rank` | Minimum LoRA rank | 4 | 1-512 |
| `max_rank` | Maximum LoRA rank | 64 | 8-512 |
| `rank_scaling` | Rank interpolation method | "linear" | linear/log/sqrt |
| `alpha_factor` | Alpha scaling factor | 2 | 1-10 |
| `w_hill` | Weight for Hill alpha in composite score | 0.7 | 0.0-1.0 |
| `w_mp_spikes` | Weight for MP spikes in composite score | 0.3 | 0.0-1.0 |

### Advanced Options

```python
# Spectral analysis parameters
spectral_analysis_results = spectral_analysis_for_adaptive_lora(
    model,
    eigenvalue_threshold=1e-5,           # Filter near-zero eigenvalues
    power_law_fitting_method="median",    # Hill estimator variant
    apply_tracy_widom_correction=True,    # Finite matrix correction
    conv_normalization=0.5                # Convolutional layer normalization
)
```

## 📊 Experimental Results

### Arithmetic Reasoning Tasks

We evaluate on four challenging arithmetic reasoning benchmarks using LLaMA-7B fine-tuned on MATH10K dataset:

| Method | Trainable Params | AQuA | GSM8K | MAWPS | SVAMP | Average |
|--------|------------------|------|-------|-------|-------|---------|
| PrefT | 0.039% | 14.2 | 24.4 | 63.4 | 38.1 | 35.0 |
| Adapter_S | 1.953% | 15.0 | 33.3 | 77.7 | 52.3 | 44.6 |
| Adapter_P | 3.542% | 18.1 | 35.3 | 82.4 | 49.6 | 46.4 |
| LoRA | 0.826% | 18.9 | 37.5 | 79.0 | 52.1 | 46.9 |
| **LoRA-AR (Ours)** | **0.821%** | **25.19** | **39.65** | **84.03** | **57.6** | **51.61** |

### Commonsense Reasoning Tasks  

Performance on eight commonsense reasoning benchmarks using LLaMA3-8B:

| Method | Trainable Params | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Average |
|--------|------------------|-------|------|------|-----------|------------|-------|-------|------|---------|
| LoRA | 0.70% | 70.8 | 85.2 | 79.9 | 91.7 | 84.3 | 84.2 | 71.2 | 79.0 | 80.8 |
| DoRA | 0.35% | 74.5 | 88.8 | 80.3 | 95.5 | 84.7 | 90.1 | 79.1 | 87.2 | 85.0 |
| DoRA | 0.71% | 74.6 | 89.3 | 79.9 | 95.5 | 85.6 | 90.5 | 80.4 | 85.8 | 85.2 |
| **LoRA-AR (Ours)** | **0.48%** | **75.01** | **88.73** | **80.6** | **95.74** | **85.87** | **90.53** | **80.71** | **86.0** | **85.4** |

### Key Achievements

- **State-of-the-art performance** on arithmetic reasoning with 4.6 point improvement over standard LoRA
- **Superior parameter efficiency** achieving best results with fewer trainable parameters
- **Consistent improvements** across diverse reasoning tasks
- **Theoretically grounded** approach eliminates heuristic rank selection

## 🔧 Advanced Usage

### Custom Spectral Analysis

```python
from src.adaptive_rank_assignment import spectral_analysis_for_adaptive_lora

# Perform detailed spectral analysis
results = spectral_analysis_for_adaptive_lora(
    model,
    eigenvalue_threshold=1e-5,
    histogram_bins=100,
    power_law_fitting_method="fix-finger",
    filter_near_zero_eigenvalues=True
)

# Access detailed metrics
for _, layer_data in results.iterrows():
    print(f"Layer: {layer_data['layer_name']}")
    print(f"Hill Alpha: {layer_data['alpha_hill']:.3f}")
    print(f"MP Spikes: {layer_data['num_spikes']}")
    print(f"Spectral Norm: {layer_data['spectral_norm']:.3f}")
```

### Integration with Existing Pipelines

```python
# Works with HuggingFace Trainer
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # Add adaptive LoRA through PEFT
)
```

## 📚 Citation

If you use Adaptive Rank LoRA in your research, please cite:

```bibtex
@article{uday2024adaptive,
  title={Fine-Tuning Large Language Models using LoRA with Adaptive Rank Allocation Based on Spectral Analysis},
  author={Paila, Uday and Pandey, Naveen and Pailla, Balakrishna and Aggarwal, Gaurav},
  year={2024},
  url={https://github.com/UdiBhaskar/Adaptive-Rank-LoRA}
}
```

## 🙏 Acknowledgments

This work builds upon several important contributions in the field:

- **[AlphaPruning](https://github.com/haiquanlu/AlphaPruning)**: For pioneering the use of Heavy-Tailed Self-Regularization theory in layer-wise optimization of neural networks
- **[WeightWatcher](https://github.com/CalculatedContent/WeightWatcher)**: For developing tools and theoretical foundations for empirical spectral density analysis of neural networks
- **[TempBalance](https://github.com/YefanZhou/TempBalance)**: For demonstrating the effectiveness of spectral analysis in adaptive learning rate scheduling

We also acknowledge:
- **Random Matrix Theory**: Mathematical foundation from Marchenko-Pastur law and Tracy-Widom statistics
- **Hill Estimator**: Power law fitting methodology for heavy-tailed distributions
- **HuggingFace**: Transformers and PEFT library ecosystem
- **TRL**: Training infrastructure and utilities for language model fine-tuning

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/UdiBhaskar/Adaptive-Rank-LoRA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/UdiBhaskar/Adaptive-Rank-LoRA/discussions)
- **Email**: udaybhaskarpaila@gmail.com

---

**Adaptive Rank LoRA**: Making parameter-efficient fine-tuning smarter, one eigenvalue at a time. 🎯