from tqdm import tqdm
import ir_datasets
import json
from collections import defaultdict,Counter
from pathlib import Path
import hashlib
import argparse
import random 
import anthropic
from datasets import load_dataset
import time


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
    queries = {}
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
            queries[query_id] = text
    else:
        with open(queries_path, "r") as f:
            for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                query_id, query_text = line.strip().split("\t")
                queries[query_id] = query_text
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

def test_create_sparse_rep(doc_text):
    min_output = 100
    tmp_spar = 'Here is the output JSON with synonyms and semantically relevant words for the given passage (300 words, weights between 0 and 1):\n\n{\n  "language_models": 1.0,\n  "neural_networks": 1.0,\n  "AI": 0.95,\n  "machine_learning": 0.95,\n  "artificial_intelligence": 0.95,\n  "deep_learning": 0.9,\n  "NLP": 0.85,\n  "natural_language_processing": 0.85,\n  "transformers": 0.8,\n  "decoder": 0.8,\n  "architecture": 0.8,\n  "RNN": 0.75,\n  "recurrent_neural_networks": 0.75, \n  "LSTM": 0.7,\n  "GRU": 0.7,\n  "state_space_models": 0.7,\n  "Mamba": 0.7,\n  "ML": 0.65,\n  "DL": 0.65,\n  "ANNs": 0.65,\n  "algorithms": 0.6,\n  "computation": 0.6,\n  "training": 0.6,\n  "inference": 0.6,\n  "parameters": 0.6,\n  "weights": 0.6,\n  "layers": 0.6,\n  "nodes": 0.6,\n  "neurons": 0.6,\n  "activation_functions": 0.55,\n  "embeddings": 0.55,\n  "representations": 0.55,\n  "sequences": 0.55,\n  "tokens": 0.55,\n  "vocabulary": 0.55,\n  "corpus": 0.55,\n  "parallel_processing": 0.5,\n  "GPU": 0.5,\n  "TPU": 0.5,\n  "hardware": 0.5,\n  "compute": 0.5,\n  "memory": 0.5,\n  "storage": 0.5,\n  "optimization": 0.45,\n  "hyperparameters": 0.45,\n  "fine-tuning": 0.45,\n  "pre-training": 0.45,\n  "supervised_learning": 0.4,\n  "unsupervised_learning": 0.4,\n  "semi-supervised_learning": 0.4,\n  "reinforcement_learning": 0.4,\n  "few-shot_learning": 0.4,\n  "zero-shot_learning": 0.4,\n  "transfer_learning": 0.4,\n  "multitask_learning": 0.4,\n  "datasets": 0.35,\n  "benchmarks": 0.35,\n  "metrics": 0.35,\n  "evaluation": 0.35,\n  "performance": 0.35,\n  "scalability": 0.35,\n  "efficiency": 0.35,\n  "generalization": 0.35,\n  "robustness": 0.35,\n  "interpretability": 0.3,\n  "explainability": 0.3,\n  "transparency": 0.3,\n  "fairness": 0.3,\n  "bias": 0.3,\n  "ethics": 0.3,\n  "privacy": 0.3,\n  "security": 0.3,\n  "reliability": 0.3,\n  "deployment": 0.25,\n  "production": 0.25,\n  "serving": 0.25,\n  "monitoring": 0.25,\n  "maintenance": 0.25,\n  "updates": 0.25,\n  "improvements": 0.25\n}'
    return tmp_spar

def create_sparse_rep(doc_text, client):
    min_output = len(text.split())
    user_text = f"Input Passage: {doc_text}.\n Output json: (at least {min_output} words - include synonyms and semantic relevant words, NOT phrases - weights in range [0,1]): "
    message = client.messages.create(
    model= "claude-3-opus-20240229",
    max_tokens=4096,
    temperature=1,
    top_p=1, 
    messages=[
    {"role": "user", "content": user_text}
    ])
    return message

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument("--queries_path", type=str)
    parser.add_argument("--collections_path", type=str)
    parser.add_argument("--query_output_path", type=str)
    parser.add_argument("--ir_path", type=str, default="irds:msmarco-passage/trec-dl-2019/judged")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    queries = read_queries(args.queries_path)
    collections = read_collection(args.collections_path)
    qid2topk = read_scoreddocs(args.ir_path)
    sorted_qid2topk = sort_qid2topk(qid2topk, args.topk)
    print(sorted_qid2topk)

    client = anthropic.Anthropic(
        api_key="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")

    with open(args.query_output_path, "w") as outfn:
        for id in tqdm(list(sorted_qid2topk), desc=args.query_output_path):
            text = queries[id]
            time.sleep(20)
            message = create_sparse_rep(text, client)
            output_json ={
                "id": id,
                "text": text,
                "generated_term_weights": message.content[0].text,
                "num_input_tokens":message.usage.input_tokens,
                "num_output_tokens":message.usage.output_tokens
            }
            outfn.write(json.dumps(output_json) + "\n")
    outfn.close()
    print("Done")