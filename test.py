import numpy as np
import gzip
import json
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
import tqdm
import gzip
import pickle

hard_negatives_scores = defaultdict(dict)
bertscore = defaultdict(dict)
rankllamascore = defaultdict(dict)

with gzip.open("/var/scratch/yzhao3/lsr_eval/data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz", "r") as f:
    data = pickle.load(f)
    for qid, scores in tqdm.tqdm(data.items(), desc="Processing CE scores"):
        for did, score in scores.items():
            hard_negatives_scores[int(qid)][int(did)] = score

with gzip.open("/var/scratch/yzhao3/lsr_eval/data/msmarco/hard_negatives_scores/rankllama-ms-marco-scores.pkl.gz", "r") as f:
    data = pickle.load(f)
    for qid, scores in tqdm.tqdm(data.items(), desc="Processing CE scores"):
        for did, score in scores.items():
            rankllamascore[int(qid)][int(did)] = score


def affine_transform_per_query(rankllama_scores, hard_negatives_scores):
    transformed_scores = defaultdict(dict)
    
    for qid in rankllama_scores:
        # Extract scores for this query
        rankllama_query_scores = np.array(list(rankllama_scores[qid].values()))
        if qid in hard_negatives_scores:
            hard_negatives_query_scores = np.array(list(hard_negatives_scores[qid].values()))
            
            # Calculate the mean and standard deviation
            rankllama_mean = np.mean(rankllama_query_scores)
            rankllama_std = np.std(rankllama_query_scores)
            hard_negatives_mean = np.mean(hard_negatives_query_scores)
            hard_negatives_std = np.std(hard_negatives_query_scores)
            
            # Compute transformation coefficients a and b
            a = hard_negatives_std / rankllama_std if rankllama_std != 0 else 0
            b = hard_negatives_mean - a * rankllama_mean
            
            # Apply the affine transformation
            for did, score in rankllama_scores[qid].items():
                transformed_scores[qid][did] = a * score + b
        else:
            # If there are no hard negatives for this qid, copy the scores unchanged
            print(qid, " not in hard negatives")
            transformed_scores[qid] = rankllama_scores[qid]
    
    return transformed_scores

transformed_rankllamascore = affine_transform_per_query(rankllamascore, hard_negatives_scores)


with gzip.open("/var/scratch/yzhao3/lsr_eval/data/msmarco/hard_negatives_scores/affine-query-rankllama-ms-marco-scores.pkl.gz", 'wb') as out_file:
    pickle.dump(transformed_rankllamascore, out_file)
