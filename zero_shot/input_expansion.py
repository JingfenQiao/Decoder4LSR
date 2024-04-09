from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from torch import nn
import ir_datasets 
import torch
import json
import argparse
import numpy as np
import pytrec_eval
import json
import ir_measures
from ir_measures import *
import ast

def clean(text):
    text = text.split('\n\n')
    if len(text) == 1:
        return text[0]
    else:
        return text[1]

def read_reps(reps_path: str):
    queries = dict()
    with open(reps_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                cleaned_data = clean(json_obj['generated_term_weights'])
                queries[json_obj['id']] = {"text":json_obj["text"] ,"generated_term_weights": json.loads(cleaned_data)}
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSONDecodeError: {e} id {id}")
    return queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument("--query_input_path", type=str)
    parser.add_argument("--query_output_path", type=str)
    args = parser.parse_args()

    queries = read_reps(args.query_input_path)
    print(queries)

    with open(args.query_output_path, "w") as outfn:
        for id, values in queries.items():
            expansion=[]
            text = values['text'].split(" ")
            for term, weight in values['generated_term_weights'].items():
                expansion.append(term)
            for i in expansion:
                if i not in text:
                    text.append(i)
            outfn.write(f"{id}\t{' '.join(text)}\n")
    outfn.close()