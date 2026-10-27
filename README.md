# Product Price Prediction with Fine-Tuned Llama 3.1 8B

**Fine-tuning an open-weight language model to estimate retail product prices from descriptions.**

This applied machine-learning project uses **QLoRA** to adapt **Llama 3.1 8B** for a specialised numerical prediction task. It covers data curation, parameter-efficient training, inference and held-out evaluation across eight categories from the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

**Published notebook result:** **$54.70 mean absolute error** on the first **250** examples from a held-out test split. In a separate comparison I ran for this task, **my fine-tuned Llama outperformed GPT-4o**. That GPT-4o comparison is a reported experiment, not a reproducible result in the published notebooks: its predictions and benchmark scores are not included in this repository.

## Results

The figures below come from [evaluation.ipynb](evaluation.ipynb), which evaluates the `improved_model_predict` function on 250 test examples. The metrics were recalculated from the notebook's saved per-example outputs.

| Metric | Fine-tuned Llama 3.1 8B |
| --- | ---: |
| Mean absolute error (MAE) | **$54.70** |
| Root mean squared logarithmic error (RMSLE) | **~0.40** |
| Absolute error below $40 | **139 / 250 (55.6%)** |
| Absolute error below $40 **or** relative error below 20% | **164 / 250 (65.6%)** |

**Metric clarification:** The original README called 65.6% "accuracy (±$40)". The evaluation code's green/hit condition is actually **absolute error < $40 OR relative error < 20%**; strictly under $40 is 55.6%. This is a regression task, so the error metrics are more informative than either threshold alone.

**GPT-4o comparison:** In my own additional experiment on this pricing task, the fine-tuned Llama performed better than GPT-4o. The GPT-4o scores, prompts and per-example outputs are not preserved here, so the comparison should be treated as a **self-reported observation**, not a documented head-to-head benchmark with publicly verifiable conditions. No numerical advantage is asserted.

These results describe the recorded sample, not expected performance on every retail category or production data. No separate human-expert comparison is provided.

## Tech stack

- **Model:** Meta Llama 3.1 8B
- **Fine-tuning:** QLoRA, PEFT, TRL's `SFTTrainer`, Hugging Face Transformers
- **Optimisation and inference:** PyTorch, bitsandbytes 4-bit quantisation, CUDA, bfloat16
- **Data:** Hugging Face Datasets, Amazon Reviews 2023, NumPy
- **Evaluation:** Python, Matplotlib and numerical error analysis
- **Experiment tracking:** Weights & Biases
- **Environment:** Jupyter notebooks / Google Colab with GPU support

## Technical implementation

### 1. Curate and balance the product data

[data_curation.ipynb](data_curation.ipynb) processes eight Amazon Reviews 2023 metadata categories:

- Automotive
- Electronics
- Office Products
- Tools and Home Improvement
- Cell Phones and Accessories
- Toys and Games
- Appliances
- Musical Instruments

The pipeline filters missing or out-of-range prices, removes selected noisy metadata and product-number patterns, and requires sufficient textual description length. Retained item text is truncated to approximately **160 tokens** before assembling the prediction prompt. Sampling reduces over-representation of cheap items and Automotive products.

The notebook records a curated sample of **408,635 products**, shuffled and split into **400,000 training examples** and **2,000 held-out test examples**. The recorded evaluation uses the first **250** held-out examples.

The model is prompted to complete a structured price answer:

```text
How much does this cost to the nearest dollar?

[cleaned product title and description]

Price is $
```

### 2. Fine-tune with QLoRA

[training.ipynb](training.ipynb) uses the following configuration:

| Parameter | Notebook setting |
| --- | --- |
| Base model | Llama 3.1 8B |
| Quantisation | 4-bit NF4, double quantisation, bfloat16 computation |
| LoRA rank / alpha | 32 / 64 |
| LoRA dropout | 0.1 |
| Adapted layers | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Epoch setting | 2 |
| Per-device batch size | 16 |
| Learning rate | 1e-4 |
| Optimiser | `paged_adamw_32bit` |
| LR schedule | Cosine, 3% warm-up |
| Maximum sequence length | 182 tokens |

A completion-only data collator trains the model to predict the price response rather than learn to reproduce the entire description. The recorded run uploaded a fine-tuned adapter to a **private** Hugging Face repository. This public repository contains notebooks and scripts, not trained model weights.

### 3. Evaluate the price estimator

[evaluation.ipynb](evaluation.ipynb) loads the base model and fine-tuned adapter. It implements:

1. A generation-based predictor that extracts the numeric price from a model continuation.
2. `improved_model_predict`, which computes a probability-weighted mean of the top three next-token candidates that can be interpreted as positive numerical prices.
3. A test harness that records price guesses, actual prices, absolute errors, squared logarithmic errors and a threshold-based classification.

The publicly saved evaluation includes 250 examples, not a comprehensive baseline or category-by-category assessment.

## Repository structure

| File | Purpose |
| --- | --- |
| [data_curation.ipynb](data_curation.ipynb) | Data exploration, filtering, sampling and split |
| [training.ipynb](training.ipynb) | Model and QLoRA training workflow |
| [evaluation.ipynb](evaluation.ipynb) | Inference, evaluation and saved example outputs |
| [items.py](items.py) | Product text cleaning, token truncation and prompts |
| [loaders.py](loaders.py) | Parallel product dataset loading |
| [requirements/](requirements/) | Dependency lists for each workflow |
| [env.example](env.example) | Example configuration variables |

## Getting started

This project was developed as a notebook workflow, including Google Colab GPU usage. Training an 8B model requires significant GPU resources, and Llama model access may require accepting conditions on Hugging Face. The original notebooks' package combinations may need adjustment in newer environments.

```bash
git clone https://github.com/8ehrad/Pricer_Agent.git
cd Pricer_Agent
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements/complete_requirements.txt
cp env.example .env
```

Use your own Hugging Face credentials and, optionally, Weights & Biases credentials. Never commit `.env` or real access tokens. These notebooks use `google.colab.userdata` for secrets, so outside Colab you must adapt credential loading.

Recommended notebook order:

1. [data_curation.ipynb](data_curation.ipynb): prepare the data.
2. [training.ipynb](training.ipynb): download Llama weights and fine-tune using a GPU.
3. [evaluation.ipynb](evaluation.ipynb): evaluate the trained adapter (requires access to the private model repository).

This is **not** a claim of one-command reproducibility without the original model permissions, GPU environment and dependencies.

## Limitations and next steps

- **Sample size:** Expand beyond 250 evaluated examples and report per-category error.
- **Benchmark documentation:** Preserve the GPT-4o benchmark prompts, predictions and scoring script in a future comparison.
- **Baselines:** Compare with simple category-level and conventional regression baselines to quantify the benefit of fine-tuning.
- **Prediction format:** Top-three next-token averaging is a task-specific heuristic, not a calibrated regression distribution.
- **Distribution:** Training data was restricted and deliberately rebalanced; this is not a representative sample of all retail products.
- **Reproducibility:** Publish an environment recipe with pinned dependencies and deterministic evaluation steps.

## Potential applications

This experiment illustrates specialised price estimation from product descriptions for tasks such as catalogue sanity checks, enrichment and anomalous price triage. It is **not** an end-to-end dynamic pricing or financial asset valuation system.

## Credits

Dataset: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). Base model: [Meta Llama 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B). The project originated in hands-on AI engineering coursework and was developed into a fine-tuning experiment.
