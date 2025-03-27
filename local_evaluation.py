# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bz2
import json
import os
import re
import logging
from datetime import datetime

import pandas as pd

from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")

from dotenv import dotenv_values

config = dotenv_values(".env")


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f, ensure_ascii=False)


def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response,
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.

    Need to handle
    Corner case 1:
        {"explanation": ...}
        Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
        {...}

    Corner case 2:
        {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        return a tuple of (explanation, score)
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        # Pattern to match the score
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens using Llama2 tokenizer"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """Helper function to create an empty batch."""
        return {
            "interaction_id": [],
            "query": [],
            "search_results": [],
            "query_time": [],
            "answer": [],
        }

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    for batch in tqdm(
        load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"
    ):
        batch_ground_truths = batch.pop(
            "answer"
        )  # Remove answers from batch and store them
        batch_predictions = participant_model.batch_generate_answer(batch)
        print(batch_predictions)
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

    return queries, ground_truths, predictions


def evaluate_predictions(
    results_df, evaluation_model_name, openai_client
):
    """
    Evaluates the predictions in the DataFrame against ground truth answers and adds evaluation columns.

    Args:
    results_df (pd.DataFrame): DataFrame containing 'queries', 'ground_truths', 'predictions' columns.
    evaluation_model_name (str): Name of the evaluation model.
    openai_client: OpenAI client instance.

    Returns:
    tuple: A tuple containing the metrics dictionary and the augmented DataFrame.
    """

    if "llama" in evaluation_model_name.lower():
        raise NotImplementedError("Llama evaluation model is not implemented yet.")

    # Initialize evaluation columns
    results_df['is_missed'] = False
    results_df['is_correct'] = False
    results_df['is_hallucination'] = False
    results_df['participant_model'] = UserModel.__name__
    results_df['evaluation_model'] = evaluation_model_name

    system_message = get_system_message()

    for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Evaluating Predictions"):
        prediction = str(row['predictions']).strip()
        query = row['queries']
        ground_truths = row['ground_truths']

        prediction_lower = prediction.lower()

        # Check if the prediction is a miss
        if 'я не знаю' in prediction_lower:
            results_df.at[idx, 'is_missed'] = True
            results_df.at[idx, 'is_correct'] = False
            results_df.at[idx, 'is_hallucination'] = False
            continue

        accuracy = -1

        # Check each ground truth
        for gt in ground_truths:
            gt = str(gt).strip()
            gt_lower = gt.lower()

            # Exact match check
            if prediction_lower == gt_lower:
                accuracy = 1
                break

            # Check for 'некоррект' in both prediction and ground truth
            pred_incorrect = 'некоррект' in prediction_lower
            gt_incorrect = 'некоррект' in gt_lower

            if pred_incorrect and gt_incorrect:
                accuracy = 1
                break
            elif pred_incorrect or gt_incorrect:
                # Mark as hallucination but continue to other ground truths
                accuracy = 0
                continue
            else:
                # Use evaluation model to determine accuracy
                messages = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"Question: {query}\nGround truth: {gt}\nPrediction: {prediction}\n",
                    },
                ]
                response = attempt_api_call(openai_client, evaluation_model_name, messages)
                if response:
                    log_response(messages, response)
                    _, acc = parse_response(response)
                    if acc == 1:
                        accuracy = 1
                        break
                    else:
                        accuracy = 0
                else:
                    # Handle API call failure, treat as incorrect
                    accuracy = 0

        # Update evaluation columns based on accuracy
        if accuracy == 1:
            results_df.at[idx, 'is_correct'] = True
            results_df.at[idx, 'is_hallucination'] = False
        else:
            results_df.at[idx, 'is_correct'] = False
            results_df.at[idx, 'is_hallucination'] = True

    # Calculate metrics
    total = len(results_df)
    n_miss = results_df['is_missed'].sum()
    n_correct = results_df['is_correct'].sum()
    n_hallucination = results_df['is_hallucination'].sum()

    # Ensure counts sum to total
    assert (n_miss + n_correct + n_hallucination) == total, "Mismatch in evaluation counts"

    score = (2 * n_correct + n_miss) / total - 1
    accuracy_rate = n_correct / total
    hallucination_rate = n_hallucination / total
    missing_rate = n_miss / total

    metrics = {
        "score": score,
        "accuracy": accuracy_rate,
        "hallucination": hallucination_rate,
        "missing": missing_rate,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n_hallucination,
        "total": total,
    }

    logger.info(metrics)

    return metrics, results_df


if __name__ == "__main__":
    from models.user_config import UserModel

    DATASET_PATH = "data/russian_crag_test.jsonl.bz2"

    # Generate predictions
    participant_model = UserModel()
    queries, ground_truths, predictions = generate_predictions(
        DATASET_PATH, participant_model
    )
    
    #results = pd.DataFrame([queries, ground_truths, predictions], columns=['queries', 'ground_truths', 'predictions'])
    results = pd.DataFrame(
        {'queries':queries,
         'ground_truths':ground_truths,
         'predictions':predictions}
    )
    
    results.to_csv('results.csv')
    
    # Evaluate Predictions
    openai_client = OpenAI(
        api_key=config["OPENAI_API_KEY"], base_url="https://openrouter.ai/api/v1"
    )
    metrics, evaluation_df = evaluate_predictions(
        results, config['OPENAI_MODEL'], openai_client
    )
    
    evaluation_df.to_csv('evaluation.csv')
