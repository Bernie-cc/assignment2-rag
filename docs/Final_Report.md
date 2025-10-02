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

## 7. Technical Report

## 8. Complete GitHub Repository
You can find all code [here](https://github.com/Bernie-cc/assignment2-rag)


