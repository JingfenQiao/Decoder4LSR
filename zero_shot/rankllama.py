import json
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
import tqdm
import gzip
import pickle
import numpy as np
import gzip

hard_negatives_scores = defaultdict(dict)
rankllamascore = defaultdict(dict)

with gzip.open("/var/scratch/tnguyen2/jingfen/lsr_eval/data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz", "r") as f:
    data = pickle.load(f)
    for qid, scores in tqdm.tqdm(data.items(), desc="Processing CE scores"):
        for did, score in scores.items():
            hard_negatives_scores[int(qid)][int(did)] = score

with gzip.open("/var/scratch/tnguyen2/jingfen/lsr_eval/data/msmarco/hard_negatives_scores/rankllama-ms-marco-scores.pkl.gz", "r") as f:
    data = pickle.load(f)
    for qid, scores in tqdm.tqdm(data.items(), desc="Processing CE scores"):
        for did, score in scores.items():
            rankllamascore[int(qid)][int(did)] = score


# Extract all scores from rankllamascore and hard_negatives_scores
rankllama_scores_list = [score for doc_scores in tqdm.tqdm(rankllamascore.values()) for score in doc_scores.values()]
hard_negatives_scores_list = [score for doc_scores in tqdm.tqdm(hard_negatives_scores.values()) for score in doc_scores.values()]

# Calculate mean and standard deviation for rankllamascore
rankllama_mean = np.mean(rankllama_scores_list)
rankllama_std = np.std(rankllama_scores_list)

# Calculate mean and standard deviation for hard_negatives_scores
hard_negatives_mean = np.mean(hard_negatives_scores_list)
hard_negatives_std = np.std(hard_negatives_scores_list)

print("rankllama mean and std", rankllama_mean, rankllama_std)
print("head_negatives mean and std", hard_negatives_mean, hard_negatives_std)

# Calculate a and b
a = hard_negatives_std / rankllama_std
b = hard_negatives_mean - a * rankllama_mean

# Applying affine transformation to rankllamascore
transformed_rankllamascore = {}
for qid, scores in rankllamascore.items():
    transformed_scores = {did: a * score + b for did, score in scores.items()}
    transformed_rankllamascore[qid] = transformed_scores

with gzip.open("/var/scratch/tnguyen2/jingfen/lsr_eval/data/msmarco/hard_negatives_scores/affine-rankllama-ms-marco-scores.pkl.gz", 'wb') as out_file:
    pickle.dump(transformed_rankllamascore, out_file)