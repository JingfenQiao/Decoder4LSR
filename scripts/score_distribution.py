from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import statistics
import tqdm
import gzip
import pickle
import os

def load_jsonl(file_path):
    hard_negatives_scores = defaultdict(dict)
    with gzip.open(file_path, "rb") as f:  # ensure binary mode for pickle
        data = pickle.load(f)
        for qid, scores in tqdm.tqdm(data.items(), desc="Processing CE scores"):
            for did, score in scores.items():
                if did in hard_negatives_scores[qid]:
                    # Properly calculate the mean if that's the intent
                    existing_score = hard_negatives_scores[qid][did]
                    hard_negatives_scores[qid][did] = (existing_score + score) / 2
                else:
                    hard_negatives_scores[qid][did] = score
    return hard_negatives_scores

MiniLM_scores = load_jsonl("/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz")
MiniLM_scores_list = [score for doc_scores in tqdm.tqdm(MiniLM_scores.values()) for score in doc_scores.values()]
MiniLM_scores=""
rankllamascore = load_jsonl("/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/hard_negatives_scores/rankllama-13b-ms-marco-scores.pkl.gz")
rankllamascore_list = [score for doc_scores in tqdm.tqdm(rankllamascore.values()) for score in doc_scores.values()]
rankllamascore=""
rankllamascore_affine = load_jsonl("/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/hard_negatives_scores/rankllama-13b-ms-marco-scores-corpus-affine.pkl.gz")
rankllamascore_affine_list = [score for doc_scores in tqdm.tqdm(rankllamascore_affine.values()) for score in doc_scores.values()]
rankllamascore_affine=""

plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 24})
sns.histplot(MiniLM_scores_list, color="blue", label="Less effective teacher (MiniLM-L-6-v2)", kde=True, stat="density", bins=30)
sns.histplot(rankllamascore_list, color="red", label="More effective teacher (RankLlama)", kde=True, stat="density", bins=30)
sns.histplot(rankllamascore_affine_list, color="green", label="More effective teacher affine transformation (RankLlama)", kde=True, stat="density", bins=30)
plt.xlabel('Relevance Score', fontsize=24)
plt.ylabel('Density', fontsize=24)
plt.legend()
plt.savefig('score_distribution_plot.png')

plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 24})
sns.histplot(MiniLM_scores_list, color="blue", label="MiniLM-L-6-v2", kde=True, stat="density", bins=30)
sns.histplot(rankllamascore_list, color="red", label="RankLlama-13B", kde=True, stat="density", bins=30)
plt.xlabel('Relevance Score', fontsize=24)
plt.ylabel('Density', fontsize=24)
plt.legend()
plt.savefig('score_distribution_plot2.png')


plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 24})
sns.histplot(rankllamascore_list, color="green", label="Before AF", kde=True, stat="density", bins=30)
sns.histplot(rankllamascore_affine_list, color="red", label="After AF", kde=True, stat="density", bins=30)
plt.xlabel('Relevance Score', fontsize=24)
plt.ylabel('Density', fontsize=24)
plt.legend()
plt.savefig('score_distribution_plot3.png')

# srun --mem=90G --time=0-12:00 --job-name=score_distribution --partition=cpu python score_distribution.py