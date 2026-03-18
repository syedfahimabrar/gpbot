# GPBot

[![Build and Push Docker Image](https://github.com/syedfahimabrar/gpbot/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/syedfahimabrar/gpbot/actions/workflows/docker-publish.yml)
[![GHCR Version](https://ghcr-badge.egpl.dev/syedfahimabrar/gpbot/latest_tag?trim=major&label=latest)](https://github.com/syedfahimabrar/gpbot/pkgs/container/gpbot)
[![GHCR Size](https://ghcr-badge.egpl.dev/syedfahimabrar/gpbot/size)](https://github.com/syedfahimabrar/gpbot/pkgs/container/gpbot)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/syedfahimabrar/gpbot/main/pyproject.toml)](https://github.com/syedfahimabrar/gpbot)
[![License](https://img.shields.io/github/license/syedfahimabrar/gpbot)](https://github.com/syedfahimabrar/gpbot)

A bilingual (English + Bangla) FAQ chatbot for Grameenphone customer service, powered by a BiLSTM intent classifier and BM25 retrieval.

**Live:** [gpbot.fahimabrar.com](https://gpbot.fahimabrar.com)

## Architecture

```
User Query (English / Bangla)
        |
        v
+-------------------+
|   Tokenizer       |  Lowercase, remove punctuation, split into words
+-------------------+
        |
        v
+-------------------+
|   BiLSTM Intent   |  Embedding -> Bidirectional LSTM -> Linear
|   Classifier      |  Predicts 1 of 14 intent categories
+-------------------+
        |
        v
  confidence >= 50%?
   /            \
  yes            no
  |               |
  v               v
+----------+   "Sorry, I couldn't
| BM25 FAQ |    understand..."
| Retrieval|
+----------+
  |
  v
Best matching FAQ answer
```

### Two-Stage Pipeline

1. **Intent Classification** - A bidirectional LSTM neural network classifies the user query into one of 14 intents (e.g., `balance_check`, `recharge`, `data_package`, `sim_replace`). Built with PyTorch.

2. **FAQ Retrieval** - The FAQ knowledge base is filtered by predicted intent, then BM25Okapi ranks questions by relevance to return the best answer.

### Supported Intents

| Intent | Description |
|--------|-------------|
| `balance_check` | Prepaid/postpaid balance inquiries |
| `recharge` | Top-up and recharge methods |
| `data_package` | Internet/data package info |
| `call_rate` | Call rate inquiries |
| `sms_package` | SMS bundle info |
| `network_issue` | Signal and connectivity problems |
| `sim_replace` | SIM replacement, 4G upgrade, eSIM |
| `myGP_app` | MyGP app usage and troubleshooting |
| `bill_payment` | Postpaid bill payment |
| `roaming` | International roaming services |
| `account_info` | Account details, number check |
| `customer_support` | Helpline, complaints |
| `offer_promotion` | Current offers and promotions |
| `number_portability` | MNP - switching operators |

## Project Structure

```
.
├── app.py              # Streamlit web UI
├── model.py            # BiLSTM model definition and text processing
├── train.py            # Training script
├── retriever.py        # BM25-based FAQ retrieval
├── data/
│   ├── intent_data.json    # ~7600 labeled training examples
│   ├── faq_kb.json         # ~250 FAQ question-answer pairs
│   └── intents.json        # Intent metadata
├── models/
│   ├── best_model.pt       # Trained model weights
│   ├── config.json         # Model hyperparameters
│   ├── word2idx.json       # Vocabulary mapping
│   └── idx2label.json      # Label mapping
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Setup

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Install Dependencies

```bash
uv sync
```

### Train the Model

```bash
uv run python train.py
```

### Run the App

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker

### Build

```bash
docker build -t gpbot .
```

### Run

```bash
docker run -p 8501:8501 gpbot
```

### Pull Pre-built Image

```bash
# Automatically picks your architecture (amd64 or arm64)
docker pull ghcr.io/syedfahimabrar/gpbot:latest

# Or specify platform explicitly
docker pull --platform linux/amd64 ghcr.io/syedfahimabrar/gpbot:latest
docker pull --platform linux/arm64 ghcr.io/syedfahimabrar/gpbot:latest

# Run
docker run -p 8501:8501 ghcr.io/syedfahimabrar/gpbot:latest
```

## Testing

See [test_questions.md](test_questions.md) for 42 sample queries across all 14 intents in both English and Bangla.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | PyTorch (BiLSTM) |
| Retrieval | rank-bm25 (BM25Okapi) |
| Frontend | Streamlit |
| Package Manager | uv |
| Container | Docker (multi-arch amd64 + arm64) |
