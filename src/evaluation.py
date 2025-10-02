'''
This module is used to evaluate the RAG system.
'''
from naive_rag import NaiveRAG
from enhanced_rag import EnhancedRAG
from utils import calculate_f1_score, calculate_exact_match

import logging
logger = logging.getLogger(__name__)
# add path to the src directory
import sys
sys.path.append("src")
import json
import time
import os
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from datasets import Dataset
import ragas
from langchain_huggingface import HuggingFacePipeline

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from ragas.llms import LangchainLLMWrapper

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import os
import numpy as np

def save_as_array(output_path, new_entry):
    # if the file exists, read the old content
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # append new entry
    data.append(new_entry)

    # overwrite and save as standard JSON array
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def evaluate_phase_1(output_path: str = "results/naive_results.json", test_data_size: int = 10):
    '''
    Evaluate the Naive RAG with k = top-1 retrieval and three different system prompts (CoT, Persona Prompting, Instruction Prompt.
    Args:
        output_path: Path to save the evaluation results. It should be a json file.
    '''
    # read config.json
    with open("config.json", "r") as f:
        config = json.load(f)
    # set top-k = 1
    config["top_k"] = 1

    # set system prompts
    system_prompts = ["cot", "persona", "instruction"]
    # evaluate for each system prompt
    for system_prompt in system_prompts:
        print(f"Evaluating {system_prompt} system prompt")
        config["prompt_type"] = system_prompt

        # write back to config.json
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)

        rag = NaiveRAG(config_path="config.json")
        rag.load_embedding_model()
        rag.load_llm_model()
        rag.load_documents("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet", col_name="passage", num_rows=3200)
        rag.load_test_data("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
        rag.build_index()
        evaluation_results = rag.evaluate(test_data_size=test_data_size, question_col="question", answer_col="answer")
        print(f"Evaluation results: {evaluation_results['num_samples']} samples, F1: {evaluation_results['average_f1_score']:.3f}, EM: {evaluation_results['average_exact_match']:.3f}")


        # save the result
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_type": system_prompt,
            "system_prompt": rag.system_prompt,
            "top_k": config["top_k"],
            "num_samples": evaluation_results["num_samples"],
            "average_f1_score": evaluation_results["average_f1_score"],
            "average_exact_match": evaluation_results["average_exact_match"]
        }

        # save the result
        save_as_array(output_path, summary)

    
def experiment(test_size = 10, output_path = "results/naive_results.json"):
    '''
    Evaluate the Naive RAG with different parameter combination of top-k (1,3,5) and prompt strategy (CoT, Persona Prompting, Instruction Prompt).
    '''

    with open("config.json", "r") as f:
        config = json.load(f)

    top_k_list = [1, 3, 5]
    prompt_strategy_list = ["cot", "persona", "instruction"]
    for top_k in top_k_list:
        for prompt_strategy in prompt_strategy_list:
            print(f"Evaluating {prompt_strategy} system prompt with top-k = {top_k}")
            config["top_k"] = top_k
            config["prompt_type"] = prompt_strategy
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
            rag = NaiveRAG(config_path="config.json")
            rag.load_embedding_model()
            rag.load_llm_model()
            rag.load_documents("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet", col_name="passage", num_rows=3200)
            rag.load_test_data("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
            rag.build_index()
            evaluation_results = rag.evaluate(test_data_size=test_size, question_col="question", answer_col="answer")
            print(f"Evaluation results: {evaluation_results['num_samples']} samples, F1: {evaluation_results['average_f1_score']:.3f}, EM: {evaluation_results['average_exact_match']:.3f}")
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_type": prompt_strategy,
                "top_k": top_k,
                "num_samples": evaluation_results["num_samples"],
                "average_f1_score": evaluation_results["average_f1_score"],
                "average_exact_match": evaluation_results["average_exact_match"]
            }
            save_as_array(output_path, summary)

def evaluate_phase_2_naive_rag(output_path: str = "results/enhanced_results.json", test_data_size: int = 100, data_path: str = None):
    """
    Evaluate the Enhanced RAG with RAGAs metrics using HuggingFace LLM.
    Args:
        output_path: Path to save the evaluation results.
        test_data_size: Number of test samples to evaluate.
        data_path: Path to the data file.
    """
    try:
        # set config and write back to config.json
        with open("config.json", "r") as f:
            config = json.load(f)
        config["prompt_type"] = "cot"
        config["top_k"] = 3
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)

        # if data_path does not exist, generate the output from Naive RAG
        if not os.path.exists(data_path):
            naive_rag = NaiveRAG(config_path="config.json")
            naive_rag.load_embedding_model()
            naive_rag.load_llm_model()
            naive_rag.load_documents("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet", col_name="passage", num_rows=3200)
            naive_rag.load_test_data("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
            naive_rag.build_index()
            evaluation_results = naive_rag.evaluate(test_data_size=test_data_size, question_col="question", answer_col="answer")

            data = {
                "user_input": [result["question"] for result in evaluation_results["detailed_results"]],
                "ground_truth": [result["ground_truth"] for result in evaluation_results["detailed_results"]],
                "response": [result["predicted"] for result in evaluation_results["detailed_results"]],
                "retrieved_contexts": [[doc["text"] for doc in result["retrieved_docs"]] 
                       for result in evaluation_results["detailed_results"]],
            }

            # save the data to a json file
            with open(data_path, "w") as f:
                json.dump(data, f, indent=4)
        # if data_path exists, load the data from the json file
        else:
            with open(data_path, "r") as f:
                data = json.load(f)
        
        # create a dataset from the data
        dataset = Dataset.from_dict(data)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # use provided API key to avoid API key error
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # read the API key from the OpenAI_API.json file
        with open("OpenAI_API.json", "r") as f:
            api_key = json.load(f)["OpenAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        # Print results
        print("RAGAs Evaluation Results (using HuggingFace LLM):")
        print("=" * 60)
        print(ragas_result)

        ragas_result_df = ragas_result.to_pandas()
        # calculate the mean of the faithfulness and answer_relevancy

        faithfulness_mean = ragas_result_df["faithfulness"].mean()
        answer_relevancy_mean = ragas_result_df["answer_relevancy"].mean()

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(dataset),
            "prompt_type": config["prompt_type"],
            "top_k": config["top_k"],
            "faithfulness": faithfulness_mean,
            "answer_relevancy": answer_relevancy_mean
        }
        save_as_array(output_path, summary)

        
    except Exception as e:
        logger.error(f"Error in evaluate_phase_2: {e}")
        print(f"Evaluation failed: {e}")
        return None
    
def evaluate_phase_2_enhanced_rag(output_path: str = "results/enhanced_results.json", test_data_size: int = 100, data_path: str = None):
    """
    Evaluate the Enhanced RAG with RAGAs metrics using HuggingFace LLM.
    Args:
        output_path: Path to save the evaluation results.
        test_data_size: Number of test samples to evaluate.
        data_path: Path to the data file.
    """
    try:
        # set config and write back to config.json
        with open("config.json", "r") as f:
            config = json.load(f)
        config["prompt_type"] = "cot"
        config["top_k"] = 3
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)

        # if data_path does not exist, generate the output from Naive RAG
        if not os.path.exists(data_path):
            enhanced_rag = EnhancedRAG(config_path="config.json")
            enhanced_rag.load_reranker()
            enhanced_rag.load_embedding_model()
            enhanced_rag.load_llm_model()
            enhanced_rag.load_documents("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet", col_name="passage", num_rows=3200)
            enhanced_rag.load_test_data("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
            enhanced_rag.build_index()
            evaluation_results = enhanced_rag.evaluate(test_data_size=test_data_size, question_col="question", answer_col="answer")

            data = {
                "user_input": [result["question"] for result in evaluation_results["detailed_results"]],
                "ground_truth": [result["ground_truth"] for result in evaluation_results["detailed_results"]],
                "response": [result["predicted"] for result in evaluation_results["detailed_results"]],
                "retrieved_contexts": [[doc["text"] for doc in result["retrieved_docs"]] 
                       for result in evaluation_results["detailed_results"]],
            }

            # save the data to a json file
            with open(data_path, "w") as f:
                json.dump(data, f, indent=4)
        # if data_path exists, load the data from the json file
        else:
            with open(data_path, "r") as f:
                data = json.load(f)
        
        # create a dataset from the data
        dataset = Dataset.from_dict(data)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # use provided API key to avoid API key error
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # read the API key from the OpenAI_API.json file
        with open("OpenAI_API.json", "r") as f:
            api_key = json.load(f)["OpenAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        # Print results
        print("Enhanced RAGAs Evaluation Results (using HuggingFace LLM):")
        print("=" * 60)
        print(ragas_result)

        ragas_result_df = ragas_result.to_pandas()
        # calculate the mean of the faithfulness and answer_relevancy

        faithfulness_mean = ragas_result_df["faithfulness"].mean()
        answer_relevancy_mean = ragas_result_df["answer_relevancy"].mean()

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(dataset),
            "prompt_type": config["prompt_type"],
            "top_k": config["top_k"],
            "faithfulness": faithfulness_mean,
            "answer_relevancy": answer_relevancy_mean
        }
        save_as_array(output_path, summary)

        
    except Exception as e:
        logger.error(f"Error in evaluate_phase_2: {e}")
        print(f"Evaluation failed: {e}")
        return None

def experiment_phase_2(test_data_size: int = 100, output_path: str = "results/enhanced_results.json"):
    """
    Evaluate the Enhanced RAG with different parameter combination of top-k (1,3,5) and prompt strategy (CoT, Persona Prompting, Instruction Prompt).
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    top_k_list = [1, 3, 5]
    prompt_strategy_list = ["cot", "persona", "instruction"]
    for top_k in top_k_list:
        for prompt_strategy in prompt_strategy_list:
            print(f"Evaluating {prompt_strategy} system prompt with top-k = {top_k}")
            config["top_k"] = top_k
            config["prompt_type"] = prompt_strategy
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
            enhanced_rag = EnhancedRAG(config_path="config.json")
            enhanced_rag.load_reranker()
            enhanced_rag.load_embedding_model()
            enhanced_rag.load_llm_model()
            enhanced_rag.load_documents("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet", col_name="passage", num_rows=3200)
            enhanced_rag.load_test_data("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
            enhanced_rag.build_index()
            evaluation_results = enhanced_rag.evaluate(test_data_size=test_data_size, question_col="question", answer_col="answer")
            print(f"Evaluation results: {evaluation_results['num_samples']} samples, F1: {evaluation_results['average_f1_score']:.3f}, EM: {evaluation_results['average_exact_match']:.3f}")
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_type": prompt_strategy,
                "top_k": top_k,
                "num_samples": evaluation_results["num_samples"],
                "average_f1_score": evaluation_results["average_f1_score"],
                "average_exact_match": evaluation_results["average_exact_match"]
            }
            save_as_array(output_path, summary)


if __name__ == "__main__":
    # evaluate_phase_1(test_data_size=100)
    # experiment(test_size=100, output_path="results/naive_results.json")
    experiment_phase_2(test_data_size=100, output_path="results/enhanced_results.json")