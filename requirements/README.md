# Requirements for Product Pricer Project

This folder contains organized requirements files for different aspects of the Product Pricer project.

## Files Overview

### `base_requirements.txt`
Core packages needed for data curation and basic functionality:
- `datasets==3.6.0` - For loading and processing datasets
- `matplotlib` - For plotting and visualization
- `huggingface_hub` - For accessing Hugging Face datasets and models
- `python-dotenv` - For loading environment variables

### `training_requirements.txt`
Packages needed for model training and fine-tuning:
- PyTorch with CUDA support (2.5.1+cu124)
- Hugging Face transformers ecosystem
- PEFT (Parameter Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning)
- BitsAndBytes for quantization
- Weights & Biases for monitoring

### `testing_requirements.txt`
Packages needed for model evaluation and testing:
- Same PyTorch and Hugging Face packages as training
- Specific versions for testing compatibility
- Additional evaluation utilities

### `development_requirements.txt`
Optional packages for development and enhanced features:
- Jupyter notebook support
- Additional data processing tools
- Enhanced visualization libraries
- Development utilities

### `complete_requirements.txt`
All packages needed to run the entire project in one file.

## Installation Instructions

### For Data Curation Only
```bash
pip install -r requirements/base_requirements.txt
```

### For Training
```bash
pip install -r requirements/training_requirements.txt
```

### For Testing/Evaluation
```bash
pip install -r requirements/testing_requirements.txt
```

### For Complete Setup
```bash
pip install -r requirements/complete_requirements.txt
```

### For Development
```bash
pip install -r requirements/base_requirements.txt
pip install -r requirements/development_requirements.txt
```

## Important Notes

1. **PyTorch Installation**: The training and testing requirements specify PyTorch with CUDA support. Install using:
   ```bash
   pip install --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Version Conflicts**: Some notebooks use slightly different versions of packages. If you encounter conflicts, use the specific requirements files for each use case.

3. **Environment Variables**: Make sure to set up your `.env` file with:
   - `HF_TOKEN` - Your Hugging Face token
   - `OPENAI_API_KEY` - Your OpenAI API key (optional)
   - `ANTHROPIC_API_KEY` - Your Anthropic API key (optional)

4. **Google Colab**: If running in Google Colab, some packages like `google-colab` are automatically available.

## Project Structure

The project consists of several notebooks:
- `data_curation1.ipynb` - Initial data exploration and curation
- `data_curation2.ipynb` - Extended dataset creation with multiple categories
- `Copy_of_Week_7_Day_3_TRAINING.ipynb` - Model training notebook
- `Copy_of_Week_7_Day_5_Testing_our_Fine_tuned_model.ipynb` - Model evaluation notebook
- `items.py` - Core Item class for data processing

Each notebook has specific requirements that are covered in the appropriate requirements files.
