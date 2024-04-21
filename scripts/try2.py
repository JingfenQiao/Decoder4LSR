<<<<<<< HEAD
import numpy as np
import gzip
import json
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
import tqdm
import gzip
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random

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
=======
import ir_datasets
import tqdm
import os

# Load the dataset
dataset = ir_datasets.load("lsr42/msmarco-passage-doct5query")
>>>>>>> a8b2635509771b18fc857e6c78f019f5468222ec

# Define the output folder and make sure it exists
output_dir = 'output_docs'
os.makedirs(output_dir, exist_ok=True)

<<<<<<< HEAD
# Calculate a and b
a = hard_negatives_std / rankllama_std
b = hard_negatives_mean - a * rankllama_mean

# Applying affine transformation to rankllamascore
transformed_rankllamascore = {}
for qid, scores in tqdm.tqdm(rankllamascore.items()):
    transformed_scores = {did: a * score + b for did, score in scores.items()}
    transformed_rankllamascore[qid] = transformed_scores

transformed_rankllamascore_list = [score for doc_scores in tqdm.tqdm(transformed_rankllamascore.values()) for score in doc_scores.values()]


min = np.min(rankllama_scores_list)
max = np.max(rankllama_scores_list)

def min_max_normalization(data, old_min, old_max, new_min=-15, new_max=15):
    """Normalize the input data array to the range [new_min, new_max]."""
    normalized = (data - old_min) / (old_max - old_min)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled

minmax_transformed_rankllamascore = {}
for qid, scores in tqdm.tqdm(rankllamascore.items()):
    transformed_scores = {did: min_max_normalization(score,old_min=min, old_max=max) for did, score in scores.items()}
    minmax_transformed_rankllamascore[qid] = transformed_scores

minmax_transformed_rankllamascore_list = [score for doc_scores in tqdm.tqdm(minmax_transformed_rankllamascore.values()) for score in doc_scores.values()]


def affine_transform_per_query(rankllama_scores, hard_negatives_scores):
    transformed_scores = defaultdict(dict)
    
    for qid in tqdm.tqdm(rankllama_scores):
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
                transformed_scores[qid][did] = float(a * score + b)
        else:
            # If there are no hard negatives for this qid, copy the scores unchanged
            print(qid, " not in hard negatives")
            transformed_scores[qid] = float(rankllama_scores[qid])
    
    return transformed_scores

query_transformed_rankllamascore = affine_transform_per_query(rankllamascore, hard_negatives_scores)
query_transformed_rankllamascore_list = [score for doc_scores in tqdm.tqdm(query_transformed_rankllamascore.values()) for score in doc_scores.values()]

log_rankllamascore = {}
for qid, scores in tqdm.tqdm(rankllamascore.items()):
    transformed_scores = {did: np.log(score + 0.001) for did, score in scores.items()}
    log_rankllamascore[qid] = transformed_scores

log_transformed_scores_list = [score for doc_scores in tqdm.tqdm(log_rankllamascore.values()) for score in doc_scores.values()]


def z_score_normalize(data, mean, std):
    return (data - mean) / std

z_score_rankllamascore = {}
for qid, scores in tqdm.tqdm(rankllamascore.items()):
    transformed_scores = {did: z_score_normalize(score, rankllama_mean, rankllama_std) for did, score in scores.items()}
    z_score_rankllamascore[qid] = transformed_scores

z_score_transformed_scores_list = [score for doc_scores in tqdm.tqdm(z_score_rankllamascore.values()) for score in doc_scores.values()]


rankllama_scores_list1 = random.sample(rankllama_scores_list, 100000)
hard_negatives_scores_list1 = random.sample(hard_negatives_scores_list, 100000)
transformed_scores_list1 = random.sample(transformed_rankllamascore_list, 100000)
query_transformed_rankllamascore_list1 = random.sample(query_transformed_rankllamascore_list, 100000)
log_transformed_rankllamascore_list1 = random.sample(log_transformed_scores_list, 100000)
z_socre_transformed_rankllamascore_list1 = random.sample(z_score_transformed_scores_list, 100000)
minmax_transformed_rankllamascore_list1 = random.sample(minmax_transformed_rankllamascore_list, 100000)

plt.figure(figsize=(12, 8))
sns.histplot(rankllama_scores_list1, color="blue", label="RankLlama Scores", kde=True, stat="density", bins=30)
sns.histplot(hard_negatives_scores_list1, color="red", label="Hard Negatives Scores", kde=True, stat="density", bins=30)
sns.histplot(query_transformed_rankllamascore_list1, color="grey", label="query Transformed RankLlama Scores", kde=True, stat="density", bins=30)
sns.histplot(transformed_scores_list1, color="green", label="linear transformation ax+b RankLlama Scores", kde=True, stat="density", bins=30)
# sns.histplot(z_socre_transformed_rankllamascore_list1, color="grey", label="z_score Transformed RankLlama Scores", kde=True, stat="density", bins=30)
# sns.histplot(log_transformed_rankllamascore_list1, color="orange", label="log Transformed RankLlama Scores", kde=True, stat="density", bins=30)
plt.title('Comparison of Score Distributions')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('score_distribution_plot3.png')

plt.figure(figsize=(12, 8))
sns.histplot(hard_negatives_scores_list1, color="red", label="Hard Negatives Scores", fill=True)
sns.histplot(query_transformed_rankllamascore_list1, color="grey", label="query Transformed RankLlama Scores", fill=True)
sns.histplot(rankllama_scores_list1, color="blue", label="RankLlama Scores", fill=True)
sns.histplot(minmax_transformed_rankllamascore_list1, color="orange", label="minmax RankLlama Scores", fill=True)
# sns.histplot(transformed_scores_list1, color="green", label="linear transformation ax+b RankLlama Scores", fill=True)
# sns.histplot(z_socre_transformed_rankllamascore_list1, color="grey", label="z_score Transformed RankLlama Scores", kde=True, stat="density", bins=30)
# sns.histplot(log_transformed_rankllamascore_list1, color="orange", label="log Transformed RankLlama Scores", kde=True, stat="density", bins=30)
plt.title('Comparison of Score Distributions')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('score_distribution_plot2.png')
=======
# Parameters for output files
max_docs_per_file = 1768365
file_count = 0
doc_count = 0

# Initialize file for writing
output_file = open(os.path.join(output_dir, f'docs_{file_count}.tsv'), 'w', encoding='utf-8')

# Iterate over each document in the dataset
for doc in tqdm.tqdm(dataset.docs_iter(), desc="Loading doc collection from ir_datasets"):
    doc_id = doc.doc_id
    text = doc.text.replace('\n', ' ').replace('\t', ' ')  # Ensure no newlines or tabs

    # Write the doc ID and text to the current file
    output_file.write(f'{doc_id}\t{text}\n')
    doc_count += 1

    # If the current file has reached its limit, start a new file
    if doc_count >= max_docs_per_file:
        output_file.close()
        file_count += 1
        doc_count = 0
        output_file = open(os.path.join(output_dir, f'docs_{file_count}.tsv'), 'w', encoding='utf-8')

# Close the last file if it was being written to
if not output_file.closed:
    output_file.close()

print(f"Documents have been processed and written into {file_count + 1} files.")
>>>>>>> a8b2635509771b18fc857e6c78f019f5468222ec
