# %%
!pip install transformers torch accelerate evaluate nltk rank-bm25 datasets sacrebleu bert_score
!pip install rouge_score

import warnings
warnings.filterwarnings('ignore')

# %%
import torch
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from evaluate import load
import nltk
from collections import Counter
import json
import time

# Download NLTK data
nltk.download('punkt', quiet=True)

# IMPORTANT: Set up Hugging Face authentication
from huggingface_hub import login

# Replace with your actual HF token from: https://huggingface.co/settings/tokens
CREDENTIALS_FILE = 'credentials.json'
try:
    with open(CREDENTIALS_FILE, 'r') as f:
        credentials = json.load(f)
    HF_TOKEN = credentials.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print(f"Error: 'HF_TOKEN' not found in {CREDENTIALS_FILE}. Please ensure the file contains a 'HF_TOKEN' key.")
except FileNotFoundError:
    print(f"Error: {CREDENTIALS_FILE} not found. Please create this file with your Hugging Face token, e.g:")
    print(f'{{ "HF_TOKEN": "<YOUR_HF_TOKEN>" }}')
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {CREDENTIALS_FILE}. Please check the file format.")

# Load evaluation metrics
rouge = load("rouge")
bleu = load("bleu")
bertscore = load("bertscore")


