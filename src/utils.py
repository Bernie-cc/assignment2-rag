"""
Utility functions for the Naive RAG system.
This module contains helper functions for data processing, embedding generation,
and evaluation metrics.
"""

import logging
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration for the RAG system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_system.log'),
            logging.StreamHandler()
        ]
    )


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from various formats (parquet, csv, json).
    
    Args:
        data_path: Path to the data file
        
    Returns:
        pandas DataFrame containing the data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    try:
        if data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            return pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        raise


def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def generate_embeddings(texts: List[str], embedding_model: HuggingFaceEmbeddings) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence-transformers.
    
    Args:
        texts: List of texts to embed
        embedding_model: HuggingFaceEmbeddings model
        
    Returns:
        numpy array of embeddings
    """
    try:
        embeddings = embedding_model.embed_documents(texts)
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise



def check_chroma_db(chroma_db_path: str) -> bool:
    """
    Check if ChromaDB data exists with data in the given path.
    """
    return os.path.exists(chroma_db_path)


def calculate_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between query and document embeddings.
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: Document embeddings matrix
        
    Returns:
        Array of similarity scores
    """
    try:
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        return similarities
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        raise


def retrieve_top_k(query_embedding: np.ndarray, doc_embeddings: np.ndarray, 
                   doc_texts: List[str], k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most similar documents for a query.
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: Document embeddings matrix
        doc_texts: List of document texts
        k: Number of top documents to retrieve
        
    Returns:
        List of dictionaries containing retrieved documents with scores
    """
    try:
        similarities = calculate_similarity(query_embedding, doc_embeddings)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'text': doc_texts[idx],
                'similarity_score': float(similarities[idx]),
                'index': int(idx)
            })
        
        logger.info(f"Retrieved top-{k} documents with scores: {[r['similarity_score'] for r in results]}")
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving top-k documents: {str(e)}")
        raise


def calculate_f1_score(predicted: str, ground_truth: str) -> float:
    """
    Calculate F1 score between predicted and ground truth answers.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    try:
        # Convert to lowercase and split into words
        pred_tokens = set(predicted.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        # Calculate precision and recall
        common_tokens = pred_tokens.intersection(gt_tokens)
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
        
    except Exception as e:
        logger.error(f"Error calculating F1 score: {str(e)}")
        return 0.0


def calculate_exact_match(predicted: str, ground_truth: str) -> float:
    """
    Calculate exact match score between predicted and ground truth answers.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    try:
        # Normalize by converting to lowercase and stripping whitespace
        pred_normalized = predicted.lower().strip()
        gt_normalized = ground_truth.lower().strip()
        
        return 1.0 if pred_normalized == gt_normalized else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating exact match: {str(e)}")
        return 0.0


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary containing results to save
        filepath: Path to save the file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {str(e)}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}. Using default config.")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration parameters.
    
    Returns:
        Dictionary containing default configuration
    """
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k": 5,
        "max_context_length": 1000,
        "temperature": 0.7,
        "max_tokens": 150,
        "log_level": "INFO"
    }


def validate_data(data: pd.DataFrame, required_columns: List[str], num_rows: int = 3200) -> bool:
    """
    Validate that the data contains required columns and is not empty.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if data.empty:
            logger.error("Data is empty")
            return False
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if len(data) != num_rows:
            logger.error(f"Data has {len(data)} rows, expected {num_rows}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return False


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents into a context string.
    
    Args:
        retrieved_docs: List of retrieved documents
        
    Returns:
        Formatted context string
    """
    try:
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            doc_text = doc['text']
            
            
            context_parts.append(f"Document {i+1}: {doc_text}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"Formatted context with {len(context_parts)} documents, length: {len(context)}")
        return context
        
    except Exception as e:
        logger.error(f"Error formatting context: {str(e)}")
        return ""
