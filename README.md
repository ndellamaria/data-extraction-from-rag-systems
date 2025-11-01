# data-extraction-from-rag-systems
CS 2881 AI Safety Assignment to recreate and extend paper: Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems

1. First, create an env
```bash
apt-get install python3.11
python -m venv env
source env/bin/activate
pip install transformers torch accelerate evaluate nltk rank-bm25 datasets sacrebleu bert_score rouge_score
touch credentials.json
```

2. Place your `credentials.json` like this
```json
{
    "HF_TOKEN": "<YOUR_HUGGINGFACE_TOKEN_HERE>"
}

