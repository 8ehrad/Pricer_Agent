# 🏷️ Product Price Prediction AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Fine-tuned Llama 3.1 8B model that predicts product prices from natural language descriptions with $54 average error.**

## 🎯 Overview

This project fine-tunes **Meta's Llama 3.1 8B** model using **QLoRA** to predict product prices from text descriptions. The system processes **400,000+ products** across 8 categories and achieves human-level accuracy.

## 🔬 Technical Implementation

### Data Pre-processing Pipeline

1. **Multi-Category Loading**: Processes 8 product categories (Electronics, Automotive, Appliances, etc.) from Amazon Reviews 2023
2. **Text Curation**: 
   - Removes irrelevant product numbers and metadata
   - Cleans and standardises descriptions
   - Truncates to 160 tokens for optimal performance
3. **Balanced Sampling**: Creates price-balanced datasets to reduce bias
4. **Quality Filtering**: Removes low-quality entries with insufficient content

### Fine-tuning Method: QLoRA

**QLoRA (Quantized Low-Rank Adaptation)** enables efficient fine-tuning with minimal memory usage:

```python
# 4-bit quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# LoRA configuration
lora_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    r=32,  # Rank
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

**Key Advantages:**
- **Memory Efficient**: 4-bit quantization reduces memory usage by 75%
- **Fast Training**: LoRA adapters train faster than full fine-tuning
- **High Quality**: Maintains model performance while reducing computational requirements

## 📈 Results & Performance

| Model | Average Error | RMSLE | Accuracy (±$40) |
|-------|---------------|-------|-----------------|
| **Fine-tuned Llama 3.1** | **$54.7** | **0.40** | **65.6%** |

### Key Achievements
- **Beats GPT-4o**: Better accuracy as state-of-the-art commercial model
- **5x Improvement**: Significantly outperforms base Llama 3.1
- **Human-Level**: Surpasses human expert performance
- **Cost-Effective**: Achieves results at fraction of API costs

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/product-price-prediction.git
cd product-price-prediction
pip install -r requirements/complete_requirements.txt
```

### Environment Setup
```bash
# Create .env file
echo "HF_TOKEN=your_huggingface_token" >> .env
echo "OPENAI_API_KEY=your_OpenAI_token" >> .env
```

## 📊 Project Structure

```
product-price-prediction/
├── data_curation.ipynb    # Data processing pipeline
├── training.ipynb         # QLoRA fine-tuning
├── evaluation.ipynb       # Model evaluation
├── items.py              # Core Item class
├── loaders.py            # Dataset utilities
└── requirements/         # Dependencies
```

## 🎯 Business Applications

- **E-commerce**: Dynamic pricing for new products
- **Retail Analytics**: Price optimization and market analysis
- **Financial Services**: Asset valuation and insurance pricing

