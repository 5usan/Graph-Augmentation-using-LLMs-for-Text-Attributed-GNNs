# Graph Augmentation using LLMs for Text-Attributed GNNs

## Project Description

**Graph Augmentation using LLMs for Text-Attributed GNNs** is a research-focused project that enhances graph data using Large Language Models (LLMs) to improve performance of Graph Neural Networks (GNNs) on text-attributed nodes. It explores how semantic insights from LLMs can be used to generate additional graph edges or features, thereby enriching the graph structure for tasks like node classification.

## Features

* **LLM-Based Graph Augmentation**: Adds edges based on LLM-inferred attributes from node texts.
* **Baseline vs. Augmented Graph Comparison**
* **Support for GCN, GAT Models**
* **Sentence-BERT Embeddings for Nodes**
* **FastAPI Server for Modular Workflow Execution**

## Installation Instructions

```bash
# Clone the repo
git clone https://github.com/5usan/Graph-Augmentation-using-LLMs-for-Text-Attributed-GNNs.git
cd Graph-Augmentation-using-LLMs-for-Text-Attributed-GNNs

# Set up Python environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** You will need [Ollama](https://ollama.com/) installed to run LLMs locally.

## Datasets

Place raw datasets in `data/raw/`:

* `twitter_dataset.csv`: Twitter user gender dataset
* `GeoTextDataset.csv`: Geolocation text dataset

## Usage Examples

### Run the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` to access interactive Swagger UI.

### API Endpoints (Example for Twitter dataset)

```bash
# Preprocess text data
curl "http://127.0.0.1:8000/pre-process_data?data_type=twitter"

# Generate graph using LLM
curl "http://127.0.0.1:8000/create_graph?data_type=twitter"

# Split graph into train/val/test
curl "http://127.0.0.1:8000/split_graph?data_type=twitter"

# Train language model (e.g. BERT)
curl "http://127.0.0.1:8000/train_eval_lm?data_type=twitter&model=bert&learning_rate=1e-5&epochs=5"

# Train GCN on augmented graph
curl "http://127.0.0.1:8000/train_eval_model?data_type=twitter&model_type=gcn"
```

## Model Details

### Language Models

* **BERT**, **DistilBERT**, **RoBERTa** fine-tuned using HuggingFace Transformers.

### Graph Neural Networks

* **GCN**: 2-layer Graph Convolutional Network
* **GAT**: 2-layer Graph Attention Network

Graphs are constructed using Sentence-BERT embeddings. LLMs classify text (e.g., gender) and connect nodes with similar text and shared LLM labels.

## Folder Structure

```
.
├── app
│   ├── main.py
│   ├── core/          # Core logic (graph, LM)
│   ├── models/        # ML models
│   ├── scripts/       # Training/evaluation
│   └── utils/         # Helpers, preprocessing
├── data
│   ├── raw/           # Raw datasets
│   ├── preprocessed/  # Cleaned CSVs
│   ├── graph_data/    # Graph .pt files
│   └── lm_data/       # Tokenized LM datasets
└── models
    ├── lm_models/     # Saved LM checkpoints
    └── graph_models/  # Saved GNN checkpoints
```

## Contribution Guidelines

* Fork and create a feature branch
* Open issues for bugs or suggestions
* Use clear commit messages and include tests if possible
* Submit PRs to the main branch for review

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

© 2025 Susan Shrestha. All rights reserved.
