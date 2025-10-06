#!/usr/bin/env python3
"""
Product Price Prediction AI - Demo Script

This script demonstrates the core functionality of the Product Price Prediction system.
It shows how to load a fine-tuned model and make price predictions on sample products.

Usage:
    python demo.py

Requirements:
    - Fine-tuned model weights (from training.ipynb)
    - All dependencies installed
"""

import os
import re
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

class ProductPricePredictor:
    """
    A class to handle product price prediction using fine-tuned Llama 3.1 model.
    """
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Meta-Llama-3.1-8B"):
        """
        Initialize the predictor with model paths.
        
        Args:
            model_path: Path to the fine-tuned model
            base_model: Base model identifier
        """
        self.base_model = base_model
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and fine-tuned model."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        print("Loading fine-tuned adapters...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        print("Model loaded successfully! ✅")
    
    def predict_price(self, product_description: str) -> float:
        """
        Predict the price of a product from its description.
        
        Args:
            product_description: Text description of the product
            
        Returns:
            Predicted price as a float
        """
        # Create the prompt
        prompt = f"How much does this cost to the nearest dollar?\n\n{product_description}\n\nPrice is $"
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=3,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract price
        price = self._extract_price(response)
        return price
    
    def _extract_price(self, text: str) -> float:
        """
        Extract price from model response.
        
        Args:
            text: Raw model response
            
        Returns:
            Extracted price as float
        """
        if "Price is $" in text:
            price_text = text.split("Price is $")[1]
            price_text = price_text.replace(',', '')
            match = re.search(r"[-+]?\d*\.\d+|\d+", price_text)
            return float(match.group()) if match else 0.0
        return 0.0
    
    def batch_predict(self, product_descriptions: List[str]) -> List[Tuple[str, float]]:
        """
        Predict prices for multiple products.
        
        Args:
            product_descriptions: List of product descriptions
            
        Returns:
            List of (description, predicted_price) tuples
        """
        results = []
        for desc in product_descriptions:
            price = self.predict_price(desc)
            results.append((desc, price))
        return results

def main():
    """Main demo function."""
    print("🏷️  Product Price Prediction AI - Demo")
    print("=" * 50)
    
    # Sample products for demonstration
    sample_products = [
        "Apple iPhone 15 Pro Max 256GB Space Black",
        "Samsung 65-inch 4K Smart TV QLED",
        "Nike Air Jordan 1 Retro High OG",
        "MacBook Pro 16-inch M2 Max 1TB SSD",
        "Tesla Model 3 Key Fob Replacement",
        "KitchenAid Stand Mixer 5 Quart",
        "Sony WH-1000XM5 Noise Canceling Headphones",
        "Dyson V15 Detect Cordless Vacuum"
    ]
    
    # Check if model exists (this would be set after training)
    model_path = "path/to/your/fine-tuned/model"  # Update this path
    
    if not os.path.exists(model_path):
        print("⚠️  Fine-tuned model not found!")
        print("Please train the model first using training.ipynb")
        print("\nFor demonstration purposes, here are some sample predictions:")
        print("-" * 50)
        
        # Mock predictions for demo
        mock_predictions = [1199, 899, 150, 2499, 150, 299, 399, 649]
        
        for product, mock_price in zip(sample_products, mock_predictions):
            print(f"Product: {product}")
            print(f"Predicted Price: ${mock_price:,.2f}")
            print("-" * 30)
    else:
        # Load model and make real predictions
        try:
            predictor = ProductPricePredictor(model_path)
            
            print("\n🎯 Making predictions on sample products...")
            print("-" * 50)
            
            results = predictor.batch_predict(sample_products)
            
            for product, price in results:
                print(f"Product: {product}")
                print(f"Predicted Price: ${price:,.2f}")
                print("-" * 30)
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please ensure the model is properly trained and the path is correct.")
    
    print("\n✨ Demo completed!")
    print("\nTo use this system:")
    print("1. Train the model using training.ipynb")
    print("2. Update the model_path in this script")
    print("3. Run: python demo.py")

if __name__ == "__main__":
    main()
