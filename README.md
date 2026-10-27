# Product Price Prediction with Fine-Tuned Llama 3.1 8B

**Fine-tuning an open-weight language model to estimate retail product prices from descriptions.**

This applied machine-learning project uses **QLoRA** to adapt **Llama 3.1 8B** for a specialised numerical prediction task. It covers data curation, parameter-efficient training, inference and held-out evaluation across eight categories from the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

On **250 held-out product descriptions**, the fine-tuned model achieved **$54.70 mean absolute error**. I also benchmarked GPT-4o on this pricing task and found that my fine-tuned Llama performed better in that comparison.

## Results

Results from [evaluation.ipynb](evaluation.ipynb), using `improved_model_predict` on 250 held-out examples:

| Metric | Fine-tuned Llama 3.1 8B |
| --- | ---: |
| Mean absolute error (MAE) | **$54.70** |
| Root mean squared logarithmic error (RMSLE) | **~0.40** |
| Absolute error below $40 | **139 / 250 (55.6%)** |
| Absolute error below $40 **or** relative error below 20% | **164 / 250 (65.6%)** |

**Prediction accuracy:** 65.6% of predictions had an absolute error below $40 **or** a relative error below 20%. Across the same sample, 55.6% were within $40 alone.

**GPT-4o comparison:** My fine-tuned Llama performed better than GPT-4o in a separate benchmark I ran for this task. The GPT-4o predictions and scores are not included in this repository.

Results are based on a sample of 250 products; performance can vary by category and price range.

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

The curated dataset contains **408,635 products**. After shuffling, **400,000 examples** were used for training and **2,000** were reserved for testing. The evaluation results above use the first **250** test examples.

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

A completion-only data collator focuses the training objective on the price response rather than reproducing the product description. The fine-tuned adapter was uploaded to a **private** Hugging Face repository; the code and notebooks are available here.

### 3. Evaluate the price estimator

[evaluation.ipynb](evaluation.ipynb) loads the base model and fine-tuned adapter. It implements:

1. A generation-based predictor that extracts the numeric price from a model continuation.
2. `improved_model_predict`, which computes a probability-weighted mean of the top three next-token candidates that can be interpreted as positive numerical prices.
3. A test harness that records price guesses, actual prices, absolute errors, squared logarithmic errors and a threshold-based classification.

The next step for evaluation would be to expand the test sample and break down errors by product category.

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

The notebooks were developed using Google Colab and GPU compute. Training requires sufficient GPU memory, Hugging Face access to Llama 3.1 8B, and compatible Python and CUDA dependencies. For newer environments, you may need to adjust the pinned package versions.

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

To reproduce the full evaluation, you will also need access to the fine-tuned adapter hosted privately on Hugging Face.

## Future improvements

- **Sample size:** Expand beyond 250 evaluated examples and report per-category error.
- **Benchmark comparison:** Repeat the GPT-4o comparison with saved prompts, predictions and a shared evaluation script.
- **Baselines:** Compare with simple category-level and conventional regression baselines to quantify the benefit of fine-tuning.
- **Prediction format:** Evaluate numerical generation and alternative aggregation methods against the current top-three next-token approach.
- **Distribution:** Check performance on natural retail price distributions in addition to the balanced evaluation data.
- **Reproducibility:** Add a runnable environment configuration and scripted evaluation.

## Potential applications

This approach could be useful for catalogue sanity checks, product-data enrichment and flagging unusually priced listings for further review.

## Credits

Dataset: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). Base model: [Meta Llama 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B). The project originated in hands-on AI engineering coursework and was developed into a fine-tuning experiment.
