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

IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"

def dot_product_rerank(query_vector, doc_vectors):
    dot_product_scores = query_vector * doc_vectors.T
    dot_product_scores = dot_product_scores.toarray()[0]
    # Get the indices of documents sorted by dot product scores (in descending order)
    sorted_indices = np.argsort(dot_product_scores)[::-1]
    return dot_product_scores, sorted_indices

def cosine_similarity_rerank(query_vector, doc_vectors):
    similarity_scores = cosine_similarity(query_vector, doc_vectors)
    sorted_indices = np.argsort(similarity_scores[0])[::-1]
    return similarity_scores, sorted_indices

def read_qrels(qrels_path: str, rel_threshold=0):
    qid2pos = defaultdict(list)  # Initialize with list
    if qrels_path.startswith(IRDS_PREFIX):
        irds_name = qrels_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for qrel in dataset.qrels_iter():
            qid, did, relevance = qrel.query_id, qrel.doc_id
            qid2pos[str(qid)].append(did)  # Correctly accumulate doc ids

def clean(text):
    text = text.split('\n\n')
    if len(text) == 1:
        return text[0]
    else:
        return text[1]

# def read_reps(reps_path: str):
#     reps = dict()
#     with open(reps_path, 'r') as file:
#         for line in file:
#             json_obj = json.loads(line)
#             id = json_obj['id']
#             reps[id] = json.loads(clean(json_obj['generated_term_weights']))
#     return reps

def read_reps(reps_path: str):
    reps = dict()
    with open(reps_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                id = json_obj['id']
                cleaned_data = clean(json_obj['generated_term_weights'])
                reps[id] = json.loads(cleaned_data)
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSONDecodeError: {e} id {id}")
    return reps

def read_scoreddocs(irds_name):
    qid2topk = defaultdict(dict)  # Changed to dict to accumulate scores
    irds_name = irds_name.replace(IRDS_PREFIX, "")
    dataset = ir_datasets.load(irds_name)
    for scoreddoc in dataset.scoreddocs_iter():
        query_id, doc_id, score = scoreddoc.query_id, scoreddoc.doc_id, scoreddoc.score
        qid2topk[query_id][doc_id] = int(score)  # Accumulate scores correctly
    return qid2topk


def sort_qid2topk(data, topk=3):
    sorted_qid2topk = {}    
    for qid, docids in data.items():
        sorted_docids = sorted(docids.items(), key=lambda x: x[1], reverse=True)[:topk]
        sorted_qid2topk[qid] = dict(sorted_docids) 
    return sorted_qid2topk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument("--query_input_path", type=str)
    parser.add_argument("--passage_input_path", type=str)
    parser.add_argument("--qrels_path", type=str, default="irds:msmarco-passage/trec-dl-2019/judged")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min_output", type=int, default=300)
    args = parser.parse_args()

    qid2topk = read_scoreddocs(args.qrels_path)
    sorted_qid2topk = sort_qid2topk(qid2topk, args.topk)
    q_reps = read_reps(args.query_input_path)
    p_reps = read_reps(args.passage_input_path)
    vec = DictVectorizer(sparse=True)
    run = defaultdict(dict)
    with open(args.output_path, "w") as outfn:
        for qid in list(sorted_qid2topk)[1:10]:
            values = sorted_qid2topk[qid]
            docs = [p_reps[docid] for docid in values]
            docids = [docid for docid in values]
            doc_vectors = vec.fit_transform(docs)
            query_vector = vec.transform([q_reps[qid]])
            dot_product_scores, sorted_indices = dot_product_rerank(query_vector, doc_vectors)
            for i in sorted_indices:
                run[qid][docids[i]] = dot_product_scores[i]
                outfn.write(f"{qid}\t{docids[i]}\t{dot_product_scores[i]}\n")
    outfn.close()

    print("start evaluation")
    qrels = ir_datasets.load(args.qrels_path.replace(IRDS_PREFIX, "")).qrels_iter()
    eval = ir_measures.calc_aggregate([RR, nDCG@10, P(rel=2)@10], qrels, run)
    print(eval)

    
        

