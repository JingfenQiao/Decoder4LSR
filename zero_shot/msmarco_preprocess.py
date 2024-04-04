from tqdm import tqdm
import ir_datasets
import json
from collections import defaultdict
from pathlib import Path
import requests
import sys
from datasets import load_dataset
import hashlib
import argparse
import random  # Added import for random

IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"

in_mem_cache = {}

def read_collection(collection_path: str, text_fields=["text"]):
    cache_key = hashlib.md5(("read_collection::" + collection_path).encode()).hexdigest()
    if cache_key in in_mem_cache:
        print(f"Previously read. Load {collection_path} from memory cache")
        return in_mem_cache[cache_key]
    doc_dict = {}
    if collection_path.startswith(IRDS_PREFIX):
        irds_name = collection_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for doc in tqdm(dataset.docs_iter(), desc=f"Loading doc collection from ir_datasets: {irds_name}"):
            doc_id = doc.doc_id
            texts = [getattr(doc, field) for field in text_fields if getattr(doc, field) is not None]
            text = " ".join(texts)
            doc_dict[doc_id] = text
    elif collection_path.startswith(HFG_PREFIX):
        hfg_name = collection_path.replace(HFG_PREFIX, "")
        dataset = load_dataset(hfg_name)
        for row in tqdm(dataset["passage"], desc=f"Loading data from HuggingFace datasets: {hfg_name}"):
            doc_dict[row["id"]] = row["text"]
    else:
        with open(collection_path, "r") as f:
            for line in tqdm(f, desc=f"Reading doc collection from {collection_path}"):
                doc_id, doc_text = line.strip().split("\t")
                doc_dict[doc_id] = doc_text
    in_mem_cache[cache_key] = doc_dict
    return doc_dict

def read_queries(queries_path: str, text_fields=["text"]):
    queries = []
    if queries_path.startswith(IRDS_PREFIX):
        irds_name = queries_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for query in tqdm(
            dataset.queries_iter(),
            desc=f"Loading queries from ir_datasets: {queries_path}",
        ):
            query_id = query.query_id
            if "wapo/v2/trec-news" in irds_name:
                doc_id = query.doc_id
                doc = dataset.docs_store().get(doc_id)
                text = doc.title + " " + doc.body
            else:
                texts = [getattr(query, field) for field in text_fields]
                text = " ".join(texts)
            queries.append([query_id, text])
    else:
        with open(queries_path, "r") as f:
            for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                query_id, query_text = line.strip().split("\t")
                queries.append([query_id, query_text])
    return queries

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


if __name__ == "__main__":  # Corrected syntax
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument('queries_path', type=str)
    parser.add_argument('collections_path', type=str)
    parser.add_argument('qrels_path', type=str)
    parser.add_argument('top100_passages_path', type=str)
    args = parser.parse_args()
    queries = read_queries(args.queries_path)
    collections = read_collection(args.collections_path)
    qrels = read_qrels(args.qrels_path)
    all_passages = set(collections.keys())
    pos_passages = set(docids for qid in qrels for docids in qrels[qid])
    neg_passages = list(all_passages - pos_passages)
    n = 698000 - len(pos_passages)
    selected_passages = random.sample(neg_passages, n)
    top100_passages = list(pos_passages) + selected_passages  

    with open(args.top100_passages_path, "w") as outfn:
        for docid in top100_passages:  # Correct variable name
            outfn.write(f"{docid}\t{collections[docid]}\n")
    
    outfn.close()