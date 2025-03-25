# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from collections import defaultdict
from typing import Any, Dict, List
from dotenv import dotenv_values

import numpy as np
import ray
import torch
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from loguru import logger
from yandex_chain import YandexLLM, YandexEmbeddings

######################################################################################################
# Configuration Parameters
######################################################################################################

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size for evaluators to call `batch_generate_answer`
SUBMISSION_BATCH_SIZE = 8

config = dotenv_values('.env')

# Yandex API configuration
YANDEX_CONFIG_PATH = "config/yandex_config.json"  # Path to your Yandex API config

######################################################################################################
# Model Implementation
######################################################################################################

class ChunkExtractor:
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(
            " ", strip=True
        )  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        print('Chunk extractor. Enter for loop')
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)
        print('Chunk extractor. Exit for loop')

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        
        print('Extract chunks. Enter for loop')
        
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"],
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]
        
        print('Extract chunks. Exit for loop')

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(
                response_ref
            )  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []
        
        print('Flatten chunks. Enter for loop')

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))
            
        print('Flatten chunks. Exit for loop')

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids


class RAGModel:
    """
    Modified RAGModel using YandexGPT and Yandex Embeddings
    """

    def __init__(self):
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        """Initialize Yandex models instead of Llama and SentenceTransformer"""
        
        print(config)
        
        if not config:
            raise Exception(
                f"Yandex API configuration file not found in .env "
                "Please provide a valid configuration file for Yandex services."
            )
        
        # Initialize Yandex Embeddings model
        self.embedding_model = YandexEmbeddings(
                                    api_key=config['YCLOUD_API_TOKEN'],
                                    folder_id = config['YCLOUD_FOLDER_ID']
                                    )
        
        # Initialize Yandex LLM
        self.llm = YandexLLM(
                    api_key=config['YCLOUD_API_TOKEN'],
                    folder_id = config['YCLOUD_FOLDER_ID']
                    )

    def calculate_embeddings(self, sentences):
        """
        Compute embeddings using Yandex Embeddings API.
        """
        # Handle empty input for both numpy arrays and regular lists
        if sentences is None or (hasattr(sentences, '__len__') and len(sentences) == 0):
            return np.array([])
            
        # Convert numpy array to list if needed
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        # Convert single sentence to list if needed
        elif isinstance(sentences, str):
            sentences = [sentences]
            
        try:
            print('Getting embeddings from Yandex')
            # Get embeddings from Yandex
            print(sentences)
            #embeddings = self.embedding_model.embed_query(sentences)
            
            embeddings = []
            
            for sentence in sentences:
                print(sentence)
                embedding = self.embedding_model.embed_document(sentence)
                embeddings.append(embedding)
            
            #embeddings = self.embedding_model.embed_documents(sentences)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error calculating embeddings: {e}")
            # Return empty embeddings with correct shape if possible
            if len(sentences) > 0:
                return np.zeros((len(sentences), 384))  # 384 is typical embedding size
            return np.array([])

    def get_batch_size(self) -> int:
        return SUBMISSION_BATCH_SIZE

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        print('Extracting chunks')
        # Chunk extraction remains the same
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        print('Calculating embeddings')
        # Calculate embeddings using Yandex
        chunk_embeddings = self.calculate_embeddings(chunks)
        query_embeddings = self.calculate_embeddings(queries)

        print('Retrieving top matches')
        # Retrieve top matches (same logic)
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_embedding = query_embeddings[_idx]
            
            relevant_chunks_mask = chunk_interaction_ids == interaction_id
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            batch_retrieval_results.append(retrieval_results)

        print('Formatting prompts')
        # Format prompts for YandexGPT
        formatted_prompts = self.format_prompts(
            queries, query_times, batch_retrieval_results
        )

        print('Generating responses')
        # Generate responses using YandexGPT
        answers = []
        for prompt in formatted_prompts:
            try:
                response = self.llm.invoke(prompt)
                answers.append(response)
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                answers.append("I don't know")
                
        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats prompts specifically for YandexGPT.
        """
        system_prompt = """You are provided with a question and references. 
        Answer the question concisely using only the provided references. 
        If the answer isn't in the references, say "I don't know"."""
        
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            # Build references section
            references = "\n".join(
                [f"- {snippet.strip()}" for snippet in retrieval_results]
            )[:MAX_CONTEXT_REFERENCES_LENGTH]

            # Format the complete prompt
            full_prompt = f"""
            {system_prompt}
            
            References:
            {references}
            
            Current Time: {query_time}
            Question: {query}
            
            Answer:
            """
            
            formatted_prompts.append(full_prompt.strip())

        return formatted_prompts
