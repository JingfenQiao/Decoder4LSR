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
import anthropic
import anthropic

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

def read_scoreddocs(irds_name):
    qid2topk = defaultdict(list)  
    dataset = ir_datasets.load(irds_name)
    for scoreddoc in dataset.scoreddocs_iter():
        query_id, doc_id, score= scoreddoc.query_id, scoreddoc.doc_id, scoreddoc.score
        if not query_id in qid2topk:
            qid2topk[query_id] = {}
        qid2topk[query_id] = {doc_id: int(score)}
    return qid2topk

def sort_qid2topk(data, topk=10):
    sorted_qid2topk = {}    
    for qid, docids in data.items():
        sorted_docids = sorted(docids.items(), key=lambda x: x[1], reverse=True)[:topk]
        sorted_qid2topk[qid] = dict(sorted_docids) 
    return sorted_qid2topk

def create_sparse_rep(doc_text, min_output):
    user_text = f"Input Passage: {doc_text}.\n Output json: (at least {min_output} words - include synonyms and semantic relevant words, NOT phrases - weights in range [0,1]): "
    message = client.messages.create(
    model= "claude-3-opus-20240229",
#   model="claude-3-sonnet-20240229",
    max_tokens=4096,
    temperature=1,
    top_p=1, 
    messages=[
    {"role": "user", "content": user_text}
    ])
    return message.content

if __name__ == "__main__":  # Corrected syntax
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument('queries_path', type=str)
    parser.add_argument('collections_path', type=str)
    parser.add_argument('qrels_path', type=str)
    parser.add_argument('passage_output_path', type=str)
    parser.add_argument('query_output_path', type=str)
    parser.add_argument('topk', type=int, default=10)
    parser.add_argument('min_output', type=int, default=300)
    args = parser.parse_args()
    queries = read_queries(args.queries_path)
    collections = read_collection(args.collections_path)
    qid2topk = read_scoreddocs("msmarco-passage/trec-dl-2019/judged"")
    sorted_qid2topk = sort_qid2topk(qid2topk, args.topk)

    client = anthropic.Anthropic(
        api_key="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",)

    with open(args.query_output_path, "w") as outfn:
        for qid, x in sorted_qid2topk.items():
            text = queries[qid]
            sparse_rep = create_sparse_rep(text, args.min_output)
            outfn.write(f"{qid}\t{sparse_rep}")
    outfn.close()

    with open(args.passage_output_path, "w") as outfn:
        for qid, x in sorted_qid2topk.items():
            text = collections[qid]
            sparse_rep = create_sparse_rep(text, args.min_output)
            outfn.write(f"{qid}\t{sparse_rep}")
    outfn.close()