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
            qid, did = qrel.query_id, qrel.doc_id
            qid2pos[str(qid)].append(did)  # Correctly accumulate doc ids
    else:
        qrels = json.load(open(qrels_path, "r"))
        for qid, doc_ids in qrels.items():  # Correctly process the loaded qrels
            qid2pos[str(qid)].extend(doc_ids)
    return qid2pos

def clean(text):
    index = text.find(':\n\n{')
    if index != -1:
        return text[index+3:]  # +3 to skip past the ":\n\n"
    else:
        return text

def read_reps(reps_path: str):
    reps = dict()
    with open(reps_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            qid = json_obj['qid']
            reps[qid] = clean(json_obj['generated_term_weights'])
    return reps

def read_scoreddocs(irds_name):
    qid2topk = defaultdict(dict)  # Changed to dict to accumulate scores
    dataset = ir_datasets.load(irds_name)
    for scoreddoc in dataset.scoreddocs_iter():
        query_id, doc_id, score = scoreddoc.query_id, scoreddoc.doc_id, scoreddoc.score
        qid2topk[query_id][doc_id] = int(score)  # Accumulate scores correctly
    return qid2topk

def sort_qid2topk(data, topk=10):
    sorted_qid2topk = {}    
    for qid, docids in data.items():
        sorted_docids = sorted(docids.items(), key=lambda x: x[1], reverse=True)[:topk]
        sorted_qid2topk[qid] = dict(sorted_docids) 
    return sorted_qid2topk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument("--query_input_path", type=str)
    parser.add_argument("--passage_input_path", type=str)
    parser.add_argument("--qrels", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--min_output", type=int, default=300)
    args = parser.parse_args()

    qrel = read_qrels(args.qrels)
    qid2topk = read_scoreddocs("msmarco-passage/trec-dl-2019/judged")
    sorted_qid2topk = sort_qid2topk(qid2topk, args.topk)
    q_reps = read_reps(args.query_input_path)
    p_reps = read_reps(args.passage_input_path)
    v = DictVectorizer(sparse=True)
    
    run = defaultdict(dict)
    with open(args.output_path, "w") as outfn:
        for qid, docids in sorted_qid2topk.items():
            query_vector = v.transform([q_reps[qid]])
            docs = [p_reps[docid] for docid in docids]
            doc_vectors = v.fit_transform(docs)
            dot_product_scores, sorted_indices = dot_product_rerank(query_vector, doc_vectors)
            
            for i in sorted_indices:
                run[qid][docids[i]] = dot_product_scores[i]
                outfn.write(f"{qid}\t{docids[i]}\t{dot_product_scores[i]}")
    outfn.close()

    print("evaluation")
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'map', 'mrr'})
    
    print(json.dumps(evaluator.evaluate(run), indent=1))
        

