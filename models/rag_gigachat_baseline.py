# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List
from dotenv import dotenv_values

import numpy as np
import ray
import torch
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from loguru import logger

from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat

from sklearn.metrics.pairwise import cosine_similarity

######################################################################################################
# Configuration Parameters
######################################################################################################

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 500
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

EMBEDDING_SIZE = 1024

WINDOW_SIZE = 3  # Sentences per chunk
OVERLAP = 1

# Batch size for evaluators to call `batch_generate_answer`
SUBMISSION_BATCH_SIZE = 8

config = dotenv_values('.env')

MODEL = config['GIGACHAT_MODEL']

# Cache configuration
EMBEDDING_CACHE_DIR = Path(".embedding_cache")
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)

######################################################################################################
# Model Implementation
######################################################################################################

class EmbeddingCache:
    @staticmethod
    def get_cache_key(text: str) -> str:
        """Generate a consistent cache key for the given text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def get_cache_path(text: str) -> Path:
        """Get the cache file path for the given text"""
        return EMBEDDING_CACHE_DIR / f"{EmbeddingCache.get_cache_key(text)}.pkl"
    
    @staticmethod
    def load_from_cache(text: str) -> np.ndarray:
        """Load embedding from cache if exists"""
        cache_path = EmbeddingCache.get_cache_path(text)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    @staticmethod
    def save_to_cache(text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_path = EmbeddingCache.get_cache_path(text)
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

class ChunkExtractor:
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """Enhanced chunk extraction with proper sentence splitting"""
        # Enhanced HTML cleaning
        soup = BeautifulSoup(html_source, "lxml")
        
        logger.info('Extracting chunks')
        
        # Remove non-content elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
            
        text = soup.get_text(" ", strip=True)
        if not text:
            return interaction_id, [""]

        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Further split long paragraphs into chunks
        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= MAX_CONTEXT_SENTENCE_LENGTH:
                chunks.append(paragraph)
            else:
                # Split paragraphs into sentences using blingfire
                sent_text, _ = text_to_sentences_and_offsets(paragraph)
                sentences = [s.strip() for s in sent_text.split('\n') if s.strip()]
                
                # Apply sliding window to sentences
                window_size = 3  # Sentences per chunk
                overlap = 1
                for i in range(0, len(sentences), window_size - overlap):
                    chunk = ' '.join(sentences[i:i+window_size])
                    chunks.append(chunk[:MAX_CONTEXT_SENTENCE_LENGTH])
        
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
        
        logger.info('Extract chunks. Enter for loop')
        
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"],
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]
        
        logger.info('Extract chunks. Exit for loop')

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
        
        logger.info('Flatten chunks. Enter for loop')

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))
            
        logger.info('Flatten chunks. Exit for loop')

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids


class RAGModel:
    """
    Modified RAGModel using SberGPT and Sber Embeddings
    """

    def __init__(self):
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        """Initialize Sber models instead of Llama and SentenceTransformer"""
        
        if not config:
            raise Exception(
                f"Sber configuration file not found in .env "
                "Please provide a valid configuration file for Sber services."
            )
        
        # Initialize Sber Embeddings model
        self.embedding_model = GigaChatEmbeddings(
                credentials=config['GIGACHAT_API_KEY'],
                scope="GIGACHAT_API_PERS",
                verify_ssl_certs=False,
            )
        
        # Initialize Sber LLM
        self.llm = GigaChat(
                credentials=config['GIGACHAT_API_KEY'],
                scope="GIGACHAT_API_PERS",
                model=MODEL,
                verify_ssl_certs=False,
            )

    def calculate_embeddings(self, sentences):
        """
        Compute embeddings using Sber Embeddings API with caching.
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
            logger.info('Getting embeddings from Sber')
            
            embeddings = []
            uncached_texts = []
            cache_indices = []
            
            # Check cache first
            for i, text in enumerate(sentences):
                cached_embedding = EmbeddingCache.load_from_cache(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    uncached_texts.append(text)
                    cache_indices.append(i)
            
            # Only compute embeddings for uncached texts
            if uncached_texts:
                logger.info(f'Computing embeddings for {len(uncached_texts)} uncached texts')
                new_embeddings = self.embedding_model.embed_documents(texts=uncached_texts)
                
                # Cache the new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    EmbeddingCache.save_to_cache(text, embedding)
                
                # Merge cached and new embeddings
                temp_embeddings = embeddings.copy()
                embeddings = []
                cached_idx = 0
                new_idx = 0
                
                for i in range(len(sentences)):
                    if i in cache_indices:
                        embeddings.append(new_embeddings[new_idx])
                        new_idx += 1
                    else:
                        embeddings.append(temp_embeddings[cached_idx])
                        cached_idx += 1
            
            logger.info('Embeddings created/loaded')
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error calculating embeddings: {e}")
            # Return empty embeddings with correct shape if possible
            if len(sentences) > 0:
                return np.zeros((len(sentences), EMBEDDING_SIZE)) 
            return np.array([])

    def get_batch_size(self) -> int:
        return SUBMISSION_BATCH_SIZE

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        self.initialize_models()
        
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        chunk_embeddings = self.calculate_embeddings(chunks)
        query_embeddings = self.calculate_embeddings(queries)

        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_embedding = query_embeddings[_idx]
            
            relevant_chunks_mask = chunk_interaction_ids == interaction_id
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Handle empty embeddings case
            if relevant_chunks_embeddings.size == 0 or query_embedding.size == 0:
                batch_retrieval_results.append(np.array([]))
                continue

            # Ensure compatible dimensions for dot product
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            # Proper cosine similarity calculation
            cosine_scores = cosine_similarity(relevant_chunks_embeddings, query_embedding)
            cosine_scores = cosine_scores.flatten()

            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            batch_retrieval_results.append(retrieval_results)

        # Format prompts and generate answers
        formatted_prompts = self.format_prompts(
            queries, query_times, batch_retrieval_results
        )

        answers = []
        for prompt in formatted_prompts:
            try:
                response = self.llm.invoke(prompt)
                answers.append(response.model_dump())
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                answers.append("I don't know")
                
        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats prompts specifically for SberGPT.
        """
        system_prompt = """Тебе дан вопрос и данные с информацией. 
        Ответь на вопрос точно, используя только предоставленные данные. 
        Если ответа нет в предоставленных данных, ответь "I don't know"."""
        
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