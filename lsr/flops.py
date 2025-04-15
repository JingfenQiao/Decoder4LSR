# Imports
import ast
import logging
import numpy as np
import os
import tqdm
from transformers import PretrainedConfig,AutoTokenizer,AutoModelForSeq2SeqLM, AutoModel
import argparse
import json
from multiprocessing import Pool
from collections import defaultdict
import threading
import multiprocessing

def combine_dicts(*dicts):
    combined = {}
    for dictionary in dicts:
        for key, value_set in dictionary.items():
            if key in combined:
                combined[key] |= value_set  # Union of sets
            else:
                combined[key] = value_set
    return combined

def loading_file(file_path):
    """Function to process a single file."""
    file_data = defaultdict(list)
    counter=0
    with open(file_path, "r") as file:
        for line in tqdm.tqdm(file,desc=file_path):
            doc = json.loads(line)
            counter+=1
            for term, _ in doc["vector"].items():
                file_data[term].append(doc["id"])

    return (file_data,counter)

def loading_data_parallel(folder_path):
    files = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file) for file in files]

    with Pool() as pool:
        (results, counter) = pool.map(loading_file, file_paths)

    results = combine_dicts(results)
    return (results, sum(counter))

def loading_data(folder_path):
    files = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file) for file in files]
    print(file_paths)

    file_data = defaultdict(list)
    nb_docs = 0
    for file_path in file_paths:
        with open(file_path, "r") as file:
            for line in tqdm.tqdm(file, desc=file_path):
                doc = json.loads(line)
                nb_docs+=1
                for term, _ in doc["vector"].items():
                    file_data[term].append(doc["id"])
    return (file_data, nb_docs)

def create_index_dist(index):
    """ Creates a distribution from the index. """
    index_dist = {}
    for k, v in index.items():
        index_dist[int(k)] = len(v)
    return index_dist

def estim_act_prob(dist, collection_size, voc_size=30522):
    """ Estimates the activation probability. """
    x = np.zeros(voc_size)
    values = list(dist.values())
    indices = [int(i) for i in dist.keys()]
    x[indices] = values
    return x / collection_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument('model_name', type=str, help='Path to the input file.')
    parser.add_argument('doc_folder', type=str, help='Path to the input file.')
    parser.add_argument('query_folder', type=str, help='Path to the output file.')
    parser.add_argument('output_floder', type=str, help='Path to the output file.')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    logger.info("Loading model...")
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    doc_index, doc_nb_docs = loading_data(args.doc_folder)
    query_index, query_nb_docs = loading_data(args.query_folder)

    # Calculation
    logger.info("Calculating FLOPs...")
    lexical_queries_index_dist = create_index_dist(query_index)
    query_index = None
    lexical_index_dist = create_index_dist(doc_index)
    doc_index = None

    p_d = estim_act_prob(lexical_index_dist, collection_size=doc_nb_docs, voc_size=tokenizer.vocab_size)
    p_q = estim_act_prob(lexical_queries_index_dist, collection_size=query_nb_docs, voc_size=tokenizer.vocab_size)
    flops = np.sum(p_d * p_q)

    res = dict(flops=flops)
    print(res)
    with open(args.output_floder, "w") as outfn:
        json.dump(res, outfn)