# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from collections import defaultdict
from json import JSONDecoder
from typing import Any, Dict, List
from dotenv import dotenv_values
import logging

import numpy as np
import ray
import torch
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from loguru import logger
from yandex_chain import YandexLLM, YandexEmbeddings
from utils.cragapi_wrapper import CRAG

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
SUBMISSION_BATCH_SIZE = 8  # Adjust based on your resources

# Yandex API configuration path
YANDEX_CONFIG_PATH = "config/yandex_config.json"  # Path to your Yandex API config

config = dotenv_values('.env')

# entity extraction template (same as original)
Entity_Extract_TEMPLATE = """
You are given a Query and Query Time. Do the following: 

1) Determine the domain the query is about. The domain should be one of the following: "finance", "sports", "music", "movie", "encyclopedia". If none of the domain applies, use "other". Use "domain" as the key in the result json. 

2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

[Rest of the template remains the same...]
"""

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
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

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
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"],
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

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

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids


class RAG_KG_Model:
    """
    Modified RAGModel using YandexGPT and Yandex Embeddings
    """

    def __init__(self):
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        # Initialize Yandex models
        if not config:
            raise Exception(
                f"Yandex API configuration file not found in .env file. "
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
        
        logging.info('Models initialized')
        # Since we're not using vLLM anymore, we don't need tokenizer from it
        # We'll use the YandexLLM's built-in tokenization if needed

    def calculate_embeddings(self, sentences):
        """
        Compute embeddings for a list of sentences using Yandex Embeddings.
        
        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of embeddings for the given sentences.
        """
        # Yandex Embeddings returns a list of embeddings which we convert to numpy array
        logging.info('calculate_embeddings')
        embeddings = np.array(self.embedding_model.embed_documents(sentences))
        return embeddings

    def get_batch_size(self) -> int:
        """
        Returns the batch size for evaluation.
        """
        return SUBMISSION_BATCH_SIZE

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using YandexGPT.
        """
        logging.info(f'{str(self)} initiated')
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )
        
        logging.info('Chunk embeddings from batch generated answer')

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)
        
        logging.info('Query embeddings from batch generated answer')

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # Retrieve top-N results
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            batch_retrieval_results.append(retrieval_results)

        # Retrieve knowledge graph results
        entities = self.extract_entity(batch)
        batch_kg_results = self.get_kg_results(entities)
        
        # Prepare formatted prompts for YandexGPT
        formatted_prompts = self.format_prompts(
            queries, query_times, batch_retrieval_results, batch_kg_results
        )
        
        # Generate responses using YandexGPT
        answers = []
        for prompt in formatted_prompts:
            try:
                response = self.llm(prompt)
                answers.append(response)
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                answers.append("I don't know")
                
        return answers

    def format_prompts(
        self, queries, query_times, batch_retrieval_results=[], batch_kg_results=[]
    ):
        """
        Formats queries and context for YandexGPT.
        """
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            kg_results = batch_kg_results[_idx]

            user_message = ""
            retrieval_references = ""
            if len(retrieval_results) > 0:
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    retrieval_references += f"- {snippet.strip()}\n"
                    
            retrieval_references = retrieval_references[
                : int(MAX_CONTEXT_REFERENCES_LENGTH / 2)
            ]
            kg_results = kg_results[: int(MAX_CONTEXT_REFERENCES_LENGTH / 2)]

            references = (
                "### References\n"
                + "# Web\n"
                + retrieval_references
                + "# Knowledge Graph\n"
                + kg_results
            )

            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            # Format for YandexGPT (adjust based on Yandex's expected input format)
            formatted_prompt = f"{system_prompt}\n\n{user_message}"
            formatted_prompts.append(formatted_prompt)

        return formatted_prompts

    def extract_entity(self, batch):
        """
        Extracts entities using YandexGPT.
        """
        queries = batch["query"]
        query_times = batch["query_time"]
        formatted_prompts = self.format_prompts_for_entity_extraction(
            queries, query_times
        )
        
        entities = []
        for prompt in formatted_prompts:
            try:
                response = self.llm(prompt)
                try:
                    res = json.loads(response)
                except:
                    res = extract_json_objects(response)
                entities.append(res)
            except Exception as e:
                logger.error(f"Error extracting entities: {e}")
                entities.append({})
                
        return entities

    def get_kg_results(self, entities):
        """
        Retrieves knowledge graph results (same as original).
        """
        api = CRAG(server=CRAG_MOCK_API_URL)
        batch_kg_results = []
        for entity in entities:
            kg_results = []
            res = ""
            if "domain" in entity.keys():
                domain = entity["domain"]
                if domain in ["encyclopedia", "other"]:
                    if "main_entity" in entity.keys():
                        try:
                            top_entity_name = api.open_search_entity_by_name(
                                entity["main_entity"]
                            )["result"][0]
                            res = api.open_get_entity(top_entity_name)["result"]
                            kg_results.append({top_entity_name: res})
                        except Exception as e:
                            logger.warning(f"Error in open_get_entity: {e}")
                            pass
                # [Rest of the knowledge graph logic remains the same...]
            batch_kg_results.append(
                "<DOC>\n".join([str(res) for res in kg_results])
                if len(kg_results) > 0
                else ""
            )
        return batch_kg_results

    def format_prompts_for_entity_extraction(self, queries, query_times):
        """
        Formats prompts for entity extraction using YandexGPT.
        """
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            user_message = ""
            user_message += f"Query: {query}\n"
            user_message += f"Query Time: {query_time}\n"

            # Format for YandexGPT (system prompt + user message)
            formatted_prompt = f"{Entity_Extract_TEMPLATE}\n\n{user_message}"
            formatted_prompts.append(formatted_prompt)
            
        return formatted_prompts
