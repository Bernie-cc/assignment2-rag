"""
Enhanced RAG (Retrieval-Augmented Generation) System Implementation.

This module implements a enhanced RAG system that:
1. Reuse all featrues from naive RAG system
2. Add query rewriting to improve the retrieval quality
3. Add retrieval-reranker to improve the final response

Author: Zijin Cui
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import os
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from utils import (
    setup_logging, load_data, clean_text, generate_embeddings,
    retrieve_top_k, calculate_f1_score, calculate_exact_match,
    save_results, load_config, get_default_config, validate_data,
    format_context
)

logger = logging.getLogger(__name__)


class EnhancedRAG:
    """
    A enhanced RAG system implementation using sentence-transformers and ChromaDB.
    It reuses all features from naive RAG system and add query rewriting and retrieval-reranker to improve the final response.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Naive RAG system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            print("Loading configuration from file")
            self.config = load_config(config_path)
        else:
            print("Using default configuration")
            self.config = get_default_config()
        
        # Setup logging
        setup_logging(self.config.get("log_level", "INFO"))
        
        # Initialize components
        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_pipeline = None
        self.chroma_client = None
        self.collection = None
        self.documents = []
        self.test_data = None
        self.system_prompt = self.load_system_prompt(self.config.get("prompt_type", "cot"))
        self.reranker = None
        self.top_k = self.config.get("top_k", 3)
        
        logger.info("Naive RAG system initialized")
    
    def load_reranker(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Load the reranker model.
        """
        try:
            self.reranker = CrossEncoder(model_name)
        except Exception as e:
            logger.error(f"Error loading reranker model: {str(e)}")
            raise
    
    def load_system_prompt(self, prompt_type: str = "cot") -> str:
        """
        Load the system prompt from the prompts.json file.
        """
        try:
            with open("prompts.json", "r") as f:
                prompts = json.load(f)
            logger.info(f"Loaded {prompt_type} type system prompt: {prompts}")
            return prompts["prompts"][prompt_type]["system_prompt"]
        except Exception as e:
            logger.error(f"Error loading system prompt: {str(e)}")
            raise
    
    def load_embedding_model(self, model_name: str = None) -> None:
        """
        Load the sentence transformer model for embeddings.
        
        Args:
            model_name: Name of the embedding model to load
        """
        try:
            model_name = model_name or self.config.get("embedding_model", "all-MiniLM-L6-v2")
            logger.info(f"Loading embedding model: {model_name}")
            
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': "cpu"},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def load_llm_model(self, model_name: str = "google/flan-t5-large") -> None:
        """
        Load the language model for text generation.
        
        Args:
            model_name: Name of the language model to load
        """
        try:
            logger.info(f"Loading LLM model: {model_name}")
            
            # Load tokenizer and model
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
            


            # Create pipeline for text generation
            self.llm_pipeline = pipeline(
                "text2text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                device="cpu",
            )
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise
    
    def load_test_data(self, data_path: str) -> None:
        """
        Load test data from a data file.
        """
        try:
            self.test_data = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")

        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def load_documents(self, data_path: str, col_name: str = "passage", num_rows: int = 3200) -> None:
        """
        Load and preprocess documents from a data file.
        
        Args:
            data_path: Path to the data file
            text_column: Name of the column containing text data
        """
        try:
            data_path = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"
            passages = pd.read_parquet(data_path)
            logger.info(f"Loading documents from {data_path}")
            
            
            # Validate data
            if not validate_data(passages, [col_name], num_rows):
                raise ValueError(f"Invalid data format. Required column: {col_name}")
            
            # Extract and clean texts
            self.documents = []
            for idx, row in passages.iterrows():
                text = clean_text(str(row[col_name]))
                if text:  # Only add non-empty texts
                    self.documents.append({
                        'id': idx,
                        'text': text,
                        'original_text': str(row[col_name])
                    })
            
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank the documents using the reranker model.

        Args:
            query: Query string
            documents: List of documents
        Returns:
            List of reranked documents
        """
        try:

            pairs = [(query, d['text']) for d in documents]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            logger.info(f"Reranked {len(ranked)} documents")
            logger.info(f"Top {self.top_k} documents will be used for generation")
            reranked_docs = [doc for doc, _ in ranked[:self.top_k]]
            assert len(reranked_docs) == self.top_k, f"Reranked {len(reranked_docs)} documents, but expected {self.top_k}"
            return reranked_docs
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            raise
    
    def build_index(self) -> None:
        """
        Build ChromaDB collection from document embeddings.
        """
        try:
            if not self.embedding_model:
                raise ValueError("Embedding model not loaded. Call load_embedding_model() first.")
            
            if not self.documents:
                raise ValueError("No documents loaded. Call load_documents() first.")
            
            logger.info("Building ChromaDB collection...")
            
            # Initialize ChromaDB persistent client
            self.chroma_client = PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            collection_name = "rag_documents"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
                # if data is already in chroma_db, skip building index
                if self.collection.count() > 0:
                    logger.info(f"Data already exists in ChromaDB, skipping build_index: {self.collection.count()} documents")
                    return
            except Exception as e:
                # Collection doesn't exist, create it
                logger.info(f"Collection doesn't exist, creating new one: {e}")
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.info(f"Created new collection: {collection_name}")
            
            # Prepare documents for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(self.documents):
                documents.append(doc['text'])
                metadatas.append({
                    'document_id': doc['id'],
                    'original_text': doc['original_text'][:1000]  # Truncate for metadata
                })
                ids.append(f"doc_{i}")
            
            # Add documents to collection
            logger.info(f"Adding {len(documents)} documents to ChromaDB...")
            
            # Add in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                batch_docs = documents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Get collection info
            collection_count = self.collection.count()
            logger.info(f"ChromaDB collection built with {collection_count} documents")
            logger.info("ChromaDB data automatically persisted to ./chroma_db")
            
        except Exception as e:
            logger.error(f"Error building ChromaDB collection: {str(e)}")
            raise
    
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query using ChromaDB.
        Since we have retrieval-reranker, we retrieve more documents than we need and then rerank them.
        Therefore, we retrieve top_k * 2 documents.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with similarity scores
        """
        try:
            if not self.collection:
                raise ValueError("Collection not built. Call build_index() first.")
            
            if not self.embedding_model:
                raise ValueError("Embedding model not loaded. Call load_embedding_model() first.")

        
            top_k = self.top_k * 2
            
            # Clean query
            clean_query = clean_text(query)
            query_embedding = self.embedding_model.embed_query(clean_query)
            
            # Search collection using ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses distance, we want similarity)
                    similarity_score = 1 - distance  # Higher similarity = lower distance
                    
                    formatted_results.append({
                        'text': doc,
                        'original_text': metadata.get('original_text', doc),
                        'similarity_score': float(similarity_score),
                        'document_id': metadata.get('document_id', f"doc_{i}")
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def generate_response(self, query: str, context: str = None, 
                         retrieved_docs: List[Dict[str, Any]] = None) -> str:
        """
        Generate a response using the language model.
        
        Args:
            query: Query string
            context: Pre-formatted context (optional)
            retrieved_docs: Retrieved documents (optional)
            
        Returns:
            Generated response
        """
        try:
            if not self.llm_pipeline:
                raise ValueError("LLM model not loaded. Call load_llm_model() first.")
            
            context = format_context(retrieved_docs)

            # Create prompt
            system_prompt = self.system_prompt
            
            prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
            # print(prompt)
            
            # Generate response
            response = self.llm_pipeline(
                prompt,
                max_length=len(prompt.split()) + self.config.get("max_tokens", 150),
                temperature=self.config.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Clean up the response
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response.
        
        Args:
            question: Question to answer
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # rewrite the query
            rewritten_query = self.rewrite_query(question)
            logger.info(f"Rewritten query: {rewritten_query}")

            # Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(rewritten_query)
    
            # Rerank the documents
            reranked_docs = self.rerank_documents(rewritten_query, retrieved_docs)

            # Generate response
            response = self.generate_response(rewritten_query, retrieved_docs=reranked_docs)
            
            # Prepare result
            result = {
                'question': question,
                'answer': response,
                'retrieved_documents': reranked_docs,
                'num_retrieved': len(retrieved_docs),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the query to improve the retrieval quality.
        """
        try:
            rewrite_prompt = f"Rewrite the query: {query} to a more specific and reconded query to improve the retrieval quality."

            response = self.llm_pipeline(
                rewrite_prompt,
                max_length=len(rewrite_prompt.split()) + self.config.get("max_tokens", 150),
                temperature=self.config.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

                        # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Clean up the response
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            raise
    
    def evaluate(self, test_data_size: int, question_col: str = "question", 
                answer_col: str = "answer") -> Dict[str, Any]:
        """
        Evaluate the RAG system on a test dataset.
        
        Args:
            test_data: DataFrame containing test questions and answers
            question_col: Name of the question column
            answer_col: Name of the answer column
            top_k: Number of documents to retrieve for each question
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            if self.test_data is None:
                raise ValueError("Test data not loaded. Call load_test_data() first.")

            if test_data_size > len(self.test_data):
                test_data_size = len(self.test_data)
                
            logger.info(f"Starting evaluation on {test_data_size} test data from total {len(self.test_data)} test samples")
            
            results = []
            f1_scores = []
            exact_matches = []

            test_data = self.test_data[:test_data_size]
            
            for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
                try:
                    question = str(row[question_col])
                    ground_truth = str(row[answer_col])
                    
                    # Get RAG response
                    rag_result = self.query(question)
                    predicted_answer = rag_result['answer']
                    
                    # Calculate metrics
                    f1 = calculate_f1_score(predicted_answer, ground_truth)
                    em = calculate_exact_match(predicted_answer, ground_truth)
                    
                    f1_scores.append(f1)
                    exact_matches.append(em)
                    
                    results.append({
                        'question': question,
                        'ground_truth': ground_truth,
                        'predicted': predicted_answer,
                        'f1_score': f1,
                        'exact_match': em,
                        'retrieved_docs': rag_result['retrieved_documents']
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing test sample {idx}: {str(e)}")
                    f1_scores.append(0.0)
                    exact_matches.append(0.0)
            
            # Calculate overall metrics
            avg_f1 = np.mean(f1_scores)
            avg_em = np.mean(exact_matches)
            
            evaluation_results = {
                'num_samples': len(test_data),
                'average_f1_score': float(avg_f1),
                'average_exact_match': float(avg_em),
                'f1_scores': f1_scores,
                'exact_matches': exact_matches,
                'detailed_results': results
            }
            
            logger.info(f"Evaluation completed. F1: {avg_f1:.3f}, EM: {avg_em:.3f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
def main():
    """
    Example usage of the Naive RAG system.
    """
    # Initialize RAG system
    rag = EnhancedRAG(config_path="config.json")
    
    # Load models
    rag.load_embedding_model()
    rag.load_llm_model()
    rag.load_reranker()
    
    # Load documents
    rag.load_documents("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet", col_name="passage", num_rows=3200)
    rag.load_test_data("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
    # Build index
    rag.build_index()
    evaluation_results = rag.evaluate(test_data_size=3, question_col="question", answer_col="answer")
    print(evaluation_results)
    
    # # Example query
    # result = rag.query(rag.test_data.iloc[3]['question'])
    # ground_truth = rag.test_data.iloc[3]['answer']
    # print(f"Question: {result['question']}")
    # print(f"Answer from RAG: {result['answer']}")
    # print(f"Ground truth: {ground_truth}")
    # print(f"F1 score: {calculate_f1_score(result['answer'], ground_truth)}")
    # print(f"Exact match: {calculate_exact_match(result['answer'], ground_truth)}")
    # print(f"Retrieved {result['num_retrieved']} documents")
    # print("Top retrieved documents:")
    # for j, doc in enumerate(result['retrieved_documents'][:5], 1):
    #     print(f"  {j}. (Score: {doc['similarity_score']:.3f}) {doc['text'][:100]}...")
    


if __name__ == "__main__":
    main()
