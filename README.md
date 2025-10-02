# Assignment2-RAG: Retrieval-Augmented Generation with Enhancements

This project implements a **Retrieval-Augmented Generation (RAG)** system with two versions:
1. **Naive RAG** – baseline system using sentence-transformer embeddings and ChromaDB for retrieval.
2. **Enhanced RAG** – extends the baseline with **query rewriting** and **document reranking** to improve retrieval and final answer quality.

The project also includes evaluation pipelines using **F1/Exact Match** and **RAGAs metrics** (faithfulness, answer relevancy, etc.).

---

## Repository Structure

```
assignment2-rag/
├── src/ # Source code for core RAG system
│ ├── naive_rag.py # Baseline (Naive) RAG pipeline
│ ├── enhanced_rag.py # Enhanced RAG with query rewriting + reranking
│ ├── evaluation.py # Evaluation with F1, EM, RAGAs metrics
│ └── utils.py # Utility functions (logging, config, helpers)
├── results/ # Experimental results
│ ├── naive_results.json # Results from Naive RAG
│ ├── enhanced_results.json # Results from Enhanced RAG
│ └── comparison_analysis.csv # Side-by-side performance comparison
├── notebooks/
│ └── data_exploration.ipynb # Dataset exploration and prototyping
├── docs/
│ └── final_report.md # Final report (summary, architecture, results)
└── requirements.txt # Python dependencies
```


---

## System Overview

### Naive RAG
- **Embeddings:** Uses `all-MiniLM-L6-v2` (384-dim) from SentenceTransformers.
- **Vector DB:** ChromaDB with cosine similarity for retrieval.
- **Pipeline:** Query embedding → Top-k retrieval → LLM answer generation.
- **Evaluation:** F1 score, Exact Match.

### Enhanced RAG
- **Query Rewriting:** Uses an LLM to rephrase queries for better retrieval.
- **Reranking:** Applies a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to reorder candidate passages.
- **Evaluation:** Adds **RAGAs metrics** – faithfulness, answer relevancy.

---

## Installation

1. Clone this repository:
```bash
   git clone https://github.com/Bernie-cc/assignment2-rag.git
   cd assignment2-rag
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. (Optional) Set your OpenAI API key for RAGAs evaluation. You should create a json file `OpenAI_API.json` and put the value of your API in the key `OpenAI_API_KEY`. The format should look like
``` json
{
   "OpenAI_API_KEY": <Replace your API Key here>
}

```

## Usage
1. Run Naive RAG
```bash
python src/naive_rag.py
```

2. Run Enhanced RAG
```bash
python src/enhanced_rag.py
```

3. Evaluate with F1 / EM (You should call `evaluate_phase_1`)
```bash
python src/evaluation.py 
```

4. Evaluate with RAGAs (You should call `evaluate_phase_2`)
```bash
python src/evaluation.py
```

## Author

- Zijin Cui
Graduate Student @ CMU MISM-BIDA
