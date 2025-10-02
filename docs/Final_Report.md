# CMU 95820 Assignment 2 Final Report

## 1. Overview
In this assignment, I implement the Naive RAG with two advanced features. I also made a complete experiment to evaluate different parameters combination. For the evaluation method, I use exact match and F1 score as basic metrics. I also use `Ragas` package to automate the evaluation process by using OpenAI API to evaluate `faithfullness` and `answer_relevancy`. 

You can find all code in this [repository](https://github.com/Bernie-cc/assignment2-rag). Please check `READMD.md` to run this project. You can find all the evaluation results in the `evaluation section` in this report.  

## 2. System Architecture

The project follows a modular architecture to separate the concerns of retrieval, augmentation, evaluation, and documentation.  
Below is the structure of the repository:

```
assignment2-rag/
├── src/ # Source code for core RAG system
│ ├── naive_rag.py # Implementation of the baseline (Naive) RAG pipeline
│ ├── enhanced_rag.py # Enhanced RAG with advanced features (query rewriting, reranking, etc.)
│ ├── evaluation.py # Evaluation logic using F1, EM, and RAGAs metrics
│ └── utils.py # Utility functions (logging, config handling, data helpers)
├── results/ # Experimental results and evaluation outputs
│ ├── naive_results.json # Evaluation results of the Naive RAG
│ ├── enhanced_results.json# Evaluation results of the Enhanced RAG
│ └── comparison_analysis.csv # Comparative performance summary
├── notebooks/ # Interactive notebooks for exploration and prototyping
│ ├── data_exploration.ipynb # Dataset exploration and initial analysis
├── docs/ # Documentation
│ ├── final_report.md # Final report (summary, architecture, results, reflection)
└── requirements.txt # Python dependencies for reproducibility
```



### Key Design Principles
- **Separation of Concerns**: Different scripts for naive vs. enhanced pipeline, evaluation, and utilities.  
- **Reproducibility**: Results are logged in JSON/CSV formats under `results/`.  
- **Experiment Traceability**: Notebooks used for exploratory analysis, final results captured in `docs/final_report.md`.  
- **Scalability**: Enhanced RAG (`enhanced_rag.py`) builds on Naive RAG by adding advanced retrieval and generation strategies.


### Workflow
1. **Data Preparation**: `data_exploration.ipynb` inspects the dataset.  
2. **Naive RAG**: Implemented in `naive_rag.py` with sentence-transformer embeddings + vector DB.  
3. **Enhanced RAG**: `enhanced_rag.py` introduces query rewriting and reranking.  
4. **Evaluation**: `evaluation.py` runs traditional (F1/EM) and RAGAs metrics.  
5. **Results Logging**: Outputs are stored in `results/`.  
6. **Reporting**: Insights and reflections compiled in `docs/final_report.md`.  

---

##  3. Dataset Setup and Exploration Report

The Mini Wikipedia dataset contains 3,200 passages stored in a single column format. Each entry represents a short textual excerpt with varying levels of detail and context. The dataset occupies approximately 1.37 MB in memory, making it lightweight and efficient for local experimentation. No missing values were identified, ensuring completeness and eliminating the need for imputation.

Exploratory analysis of text length revealed substantial variation. Passage lengths range from a single word to over 2,500 characters, with a mean of 390 characters and about 62 words on average. The distribution shows a large spread, with many passages being concise definitions or descriptions, while others are extended paragraphs. Sentence counts range from 1 to 32, with a median of 4, indicating that most passages provide compact but coherent information units. This variability suggests the need for embedding models that can handle both short factual entries and longer narrative passages.

Sample entries confirm the dataset’s encyclopedic style. For example, the Uruguay passages describe geography, demographics, and historical background. Such passages are well-suited for retrieval-based tasks since they contain fact-rich information with a clear topic focus.

For the retrieval-augmented generation pipeline, I will use the **all-MiniLM-L6-v2** embedding model to encode passages due to its efficiency and strong performance on semantic similarity tasks. These embeddings will be stored in **FAISS** for vector search. On the generation side, I plan to use **google/flan-t5-large** via hugging face API calls as the large language model. This setup balances reproducibility, accessibility, and performance.

To validate the pipeline, I will use the test set provided along with **rag-mini-wikipedia**, such as “Was Abraham Lincoln the sixteenth President of the United States?” or “Did Lincoln sign the National Banking Act of 1863?” Each query has a ground truth answer, enabling systematic evaluation in later phases.

## 4. Naive RAG System Implementation Report

### Embedding Generation with Sentence-Transformers

The system utilizes the recommended "all-MiniLM-L6-v2" model from sentence-transformers for generating high-quality document and query embeddings. This model provides 384-dimensional vector representations optimized for semantic similarity tasks.

### Vector Database with ChromaDB
Since I previoudly use ChroomaDB as vector DB to build the RAG, I still use it this time instead of recommended FAISS. I use ChromaDB to generate a persistent storage with automatic data persistence to disk, preventing duplicating indexing. I also use cosine similarity for efficient approximate nearest neighbor search. 

### Retrieval and Response Generation
The system implements a complete RAG pipeline combining retrieval and generation. Once it get the query from user, it will use same emebedding model to embed the query and then use embedding vector to search in ChromaDB to get the similar chunks. Then the system prompt, original query and retrived chunks will be combined together and input in LLM and get the result.  

### Evaluation
The current RAG system also support evaluation based on F1 score and exact match score. The detailed implementation and result can be found in evaluation phase report. 

## 5. Enhanced RAG System Implementation Report

### Query Rewriting for Improved Retrieval
The Enhanced RAG system introduces **query rewriting** as a preprocessing step before document retrieval.  
Instead of directly using the raw user query, the system leverages an LLM to rephrase or expand the query into a more specific and retrieval-friendly form.  

This mechanism reduces ambiguity and improves recall of relevant passages by aligning the query better with the style of the indexed documents.  
For example, vague user inputs such as *"Lincoln banking law"* are rewritten into *"Did Abraham Lincoln sign the National Banking Act of 1863?"*, resulting in improved retrieval accuracy.



### Document Reranking with Cross-Encoder
After retrieving an initial set of candidate passages (top-`k * 2`), the Enhanced RAG applies a **Cross-Encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).  

- Each candidate passage is paired with the rewritten query.
- The reranker computes semantic relevance scores at a finer granularity compared to embedding similarity.
- The final top-`k` passages are selected based on reranker scores and passed to the generator.

This step helps reduce noise from approximate vector search and ensures that the most semantically relevant contexts are used for answer generation, leading to higher **faithfulness** and **answer relevancy**.



### Summary
Compared to the **Naive RAG system**, the Enhanced RAG improves both **retrieval quality** and **response generation accuracy** by:
1. Rewriting queries for better alignment with indexed documents.
2. Reranking retrieved passages with a Cross-Encoder to refine the evidence set.  

These enhancements are lightweight extensions on top of the Naive RAG pipeline but yield significant gains in **retrieval precision** and overall **answer quality**.


## 6. Evaluation Report

### Evaluation Methodology
To assess the performance of both the **Naive RAG** and **Enhanced RAG** systems, we adopted a consistent evaluation framework using **100 queries per run** across diverse knowledge-intensive tasks. The evaluation utilized both **traditional accuracy-based metrics** and **RAG-specific quality measures**:

- **Average F1 Score**: Measures token overlap between predicted and ground truth answers. Sensitive to partial correctness.  
- **Exact Match (EM)**: Binary metric checking whether the predicted answer exactly matches the ground truth.  
- **Faithfulness (RAGAs)**: Evaluates whether the generated answers are grounded in the retrieved context.  
- **Answer Relevancy (RAGAs)**: Measures semantic alignment between predicted responses and user queries.  

We varied two key experimental dimensions:  
- **Prompt Type**: Chain-of-Thought (CoT), Persona Prompting, and Instruction Prompting.  
- **Top-k Retrieval**: Number of documents retrieved (k = 1, 3, 5).  

A notable limitation emerged: **F1 and EM metrics often underestimate performance** when models provide explanatory responses alongside concise answers. For instance, if the gold answer is `1861–1865` but the model outputs *“The American Civil War was fought from 1861 to 1865 between Union and Confederacy”*, the system may be penalized despite being factually correct.

---

### Experimental Results

#### Quantitative Comparison (F1 and EM)

| Prompt Type   | Top-k | Naive RAG F1 | Naive RAG EM | Enhanced RAG F1 | Enhanced RAG EM |
|---------------|-------|--------------|--------------|-----------------|-----------------|
| **CoT**       | 1     | 0.09        | 0.03         | 0.157           | 0.05            |
|               | 3     | 0.143        | 0.04         | 0.180           | 0.07            |
|               | 5     | 0.230        | 0.10         | 0.185           | 0.06            |
| **Persona**   | 1     | 0.458        | 0.41         | 0.519           | 0.44            |
|               | 3     | 0.518        | 0.46         | 0.545           | 0.47            |
|               | 5     | 0.543        | 0.45         | 0.575           | 0.53            |
| **Instruction** | 1   | 0.35        | 0.31         | 0.469           | 0.41            |
|               | 3     | 0.481        | 0.39         | 0.595           | 0.53            |
|               | 5     | 0.556        | 0.46         | 0.560           | 0.48            |

---

#### Qualitative Metrics (Faithfulness and Answer Relevancy)

| System        | Prompt | Top-k | Faithfulness | Answer Relevancy |
|---------------|--------|-------|--------------|------------------|
| **Naive RAG** | CoT    | 3     | 0.748        | 0.844            |
| **Enhanced RAG** | CoT | 3     | 0.782        | 0.824            |

---

### Key Observations

1. **Prompt Type Sensitivity**  
   Persona and Instruction prompts consistently outperform CoT. The concise-answer bias in evaluation metrics penalizes CoT since it tends to produce verbose answers.  

2. **Impact of Retrieval Depth (k)**  
   - Extremely low retrieval (k=1) harms performance since the retrieved chunk may lack direct answer coverage.  
   - Increasing k generally improves performance, but gains plateau between k=3 and k=5, and in some CoT cases even degrade due to noise from irrelevant documents.  

3. **Naive vs Enhanced**  
   Enhanced RAG improves **F1/EM scores by 5–10% relative** under Persona and Instruction prompts, showing the effectiveness of **query rewriting and reranking**. Gains are less clear in CoT, highlighting that output verbosity mismatches with F1/EM evaluation.

---

### Enhancement Analysis

The **Enhanced RAG** incorporated two additional components:  

1. **Query Rewriting**: Reformulating vague or underspecified user questions into more precise forms before retrieval.  
   - **Effectiveness**: Helped improve retrieval accuracy, especially for queries where wording mismatches with document phrasing.  
   - **Challenge**: Risk of over-specification, where rewritten queries inadvertently narrow context and exclude relevant documents.  

2. **Reranking (Cross-Encoder)**: Reordering retrieved chunks based on semantic similarity to the query.  
   - **Effectiveness**: Reduced noise when k > 1, ensuring the top 3–5 contexts were highly relevant.  
   - **Observation**: Reranking had limited effect when k=1, since retrieval quality is bottlenecked by initial vector search.  

#### Performance Interpretation
- **Faithfulness** improved slightly (from 0.748 → 0.782), suggesting reranking helps ensure answers stay grounded in retrieved evidence.  
- **Answer Relevancy** remained high (~0.82), showing stable alignment across systems.  
- **F1 and EM** saw the largest improvements under Persona and Instruction prompts, validating that enhancements work best when the model is encouraged to output short, context-focused answers.  

---

### Failure Analysis

- **Verbose Outputs**: CoT often diluted EM/F1 due to unnecessary reasoning in responses.  
- **Low-k Limitations**: Single-document retrieval frequently missed crucial evidence.  
- **Noise at Higher k**: At k=5, irrelevant documents sometimes distracted the LLM despite reranking, reducing benefits.  

---

### Summary
The evaluation demonstrates that **Enhanced RAG outperforms Naive RAG in most prompt and retrieval settings**, particularly under **Persona and Instruction prompting with top-k = 3–5**. While CoT prompts underperform on token-level metrics, qualitative faithfulness scores indicate that reasoning-driven answers are still reliable but penalized by surface-level evaluation.  

Overall, **query rewriting and reranking proved effective enhancements**, though future work should explore:  
- Better evaluation beyond F1/EM to capture correctness in verbose answers.  
- Adaptive top-k selection based on query type.  
- Mitigating noise at higher k through smarter context pruning.  


## 7. Technical Report
### Executive Summary 
Please refer to Section `Overview` 

### System Architecture
Please refer to section `System Architecture`

### Experimental Results 
Please refer to sectio `Evaluation Report`

### Enhancement Analysis
Please refer to sectio `Evaluation Report`

### Production Considerations

While the Enhanced RAG system is designed for experimentation and evaluation within an academic setting, several considerations would be essential for production deployment. **Scalability** is the foremost concern: both the embedding database (ChromaDB) and the reranker model would need to handle larger corpora and higher query throughput. In a real-world scenario, ChromaDB could be replaced or scaled with distributed vector databases (e.g., Milvus, Pinecone, FAISS sharding) to accommodate millions of documents. The LLM components (query rewriting, generation) should be containerized and deployed with inference servers such as **Ray Serve** or **TorchServe**, supporting GPU acceleration and horizontal scaling.

**Deployment recommendations** include separating system components into microservices: (1) embedding and indexing service, (2) retrieval service, (3) reranking service, and (4) generation service. This modular approach would allow independent scaling and fault isolation. Caching frequently asked queries, batch processing embeddings, and adopting approximate nearest neighbor (ANN) search algorithms are practical optimizations. Monitoring pipelines with metrics such as latency, throughput, and hallucination rates would also be critical in production.

However, there are **limitations**. Query rewriting introduces variability, which may degrade performance if the LLM produces irrelevant reformulations. The reranker (cross-encoder) is more computationally expensive than a bi-encoder retriever, limiting real-time performance for high-volume applications. Additionally, the evaluation framework (RAGAs) currently relies on external LLM calls, which can be costly and non-deterministic. Thus, while this system demonstrates improved retrieval quality and answer faithfulness in a controlled environment, significant engineering effort is required for robust, enterprise-grade deployment.


## 8. Appendices

### AI Usage Log

- **Tool**: ChatGPT (GPT-5, OpenAI), HuggingFace Transformers 4.x, Sentence-Transformers 2.x, Ragas 0.3.5  
- **Purpose**: Debugging Python errors, refining evaluation pipeline, documenting architecture and methodology.  
- **Input**: Queries included error traces (e.g., `AttributeError: 'dict' object has no attribute 'strip'`), requests for documentation sections, and clarification on RAGAs metrics.  
- **Output Usage**: AI suggestions informed bug fixes (e.g., dict unpacking in reranker), produced draft documentation (e.g., README.md, Production Considerations), and provided explanations of metric requirements.  
- **Verification**: All AI outputs were manually reviewed, tested in the local environment, and cross-referenced with official library documentation (ChromaDB, Ragas, HuggingFace).

### Technical Specifications
- **Language**: Python 3.12  
- **Dependencies**: See `requirements.txt` (core: `transformers`, `sentence-transformers`, `chromadb`, `ragas`, `langchain-huggingface`).  
- **Models**:  
  - Embeddings: `all-MiniLM-L6-v2` (384-dim)  
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`  
  - LLM: `google/flan-t5-large` for rewriting and generation  
- **Hardware**: MacBook Pro M4 (CPU + Apple Silicon GPU via `mps` backend)  

### Reproducibility Instructions
1. Clone repository and install dependencies via `pip install -r requirements.txt`.  
2. Run `src/naive_rag.py` or `src/enhanced_rag.py` to build indexes and test queries.  
3. Evaluate models with `src/evaluation.py` using both F1/EM metrics and RAGAs.  
4. To reproduce exact results, use the provided Mini Wikipedia dataset from HuggingFace (`rag-mini-wikipedia`).  
5. Results are logged automatically in `results/` for audit and comparison.  



###  Academic Integrity Notes
- All **core RAG components** (index building, retrieval, reranking, evaluation logic) were independently implemented.  
- AI assistance was restricted to **debugging, documentation drafting, and conceptual clarification**.  
- Performance analysis and interpretation were conducted manually, ensuring independent reasoning and understanding of results.  


## 9. Complete GitHub Repository
You can find all code [here](https://github.com/Bernie-cc/assignment2-rag)


