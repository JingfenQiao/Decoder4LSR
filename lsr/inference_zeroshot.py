from lsr.models import DualSparseEncoder
from lsr.tokenizer import Tokenizer
import torch
from tqdm import tqdm
from pathlib import Path
import os
from collections import Counter
import json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import time
import datetime
import logging
import ir_datasets
from datasets import load_dataset
from nltk import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"

def write_to_file(f, result, type):
    if type == "query":
        rep_text = " ".join(Counter(result["vector"]).elements()).strip()
        if len(rep_text) > 0:
            f.write(f"{result['id']}\t{rep_text}\n")
    else:
        f.write(json.dumps(result) + "\n")

def get_token_ids_in_text(text, tokenizer, nltk_tokenize=False):
    if nltk_tokenize:    
        words = [i for i in word_tokenize(text.lower()) if i.lower() not in stopwords.words('english')]
        token_ids_in_text = tokenizer(" ".join(words)).input_ids
    else:        
        token_ids_in_text = tokenizer(text.lower()).input_ids
    return token_ids_in_text, len(token_ids_in_text)

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def inference(cfg: DictConfig,):
    print(OmegaConf.to_container(cfg.inference_arguments, resolve=True))
    wandb.init(
        mode="disabled",
        project=cfg.wandb.setup.project,
        group=cfg.exp_name,
        job_type="inference",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    cfg = cfg.inference_arguments
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(cfg.output_dir).joinpath(cfg.output_file)
    file_writer = open(output_path, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(msg=f"Running inference on {device}",level=1)
    logger.log(msg=f"Loading model from {cfg.model_path}",level=1)
    model = DualSparseEncoder.from_pretrained(cfg.model_path)
    model.eval()
    model.to(device)
    tokenizer_path = os.path.join(cfg.model_path, "tokenizer")
    logger.log(msg=f"Loading tokenizer from {tokenizer_path}",level=1)
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    ids = []
    texts = []
    if cfg.type == "query":
        # prompt = ""
        prompt = "<s>[INST] You are a powerful lexical search engine. You are tasked with generating important terms that can represents the semantic relevance of query in retrieval task. Make sure your word is in lowercase.\n Query: "
    else:
        # prompt = ""
        prompt = "<s>[INST] You are a powerful lexical search engine. You are tasked with generating important terms that can represents the semantic relevance of passage in retrieval task. Make sure your word is in lowercase. \n Passage: "
    
    if cfg.input_format in ("tsv","json"):
        with open(cfg.input_path, "r") as f:
            if cfg.input_format == "tsv":
                for line in tqdm(f, desc=f"Reading data from {cfg.input_path}"):
                    try:
                        idx, text = line.strip().split("\t")
                        text = prompt + text
                        ids.append(idx)
                        texts.append(text)
                    except:
                        pass
            elif cfg.input_format == "json":
                for line in tqdm(f, desc=f"Reading data from {cfg.input_path}"):
                    line = json.loads(line.strip())
                    idx = line["_id"]
                    if "title" in line:
                        text = (line["title"] + " " + line["text"]).strip()
                    else:
                        text = line["text"].strip()
                    text = prompt + text
                    ids.append(idx)
                    texts.append(text)
    elif cfg.input_path.startswith(HFG_PREFIX):
        hfg_name = cfg.input_path.replace(HFG_PREFIX, "")
        dataset = load_dataset(hfg_name)
        for doc in tqdm(dataset["passage"], desc=f"Reading data from ir_datasets {cfg.input_path}"):
            idx = doc["id"]
            text = (doc["text"]).strip()
            text = prompt + text
            ids.append(idx)
            texts.append(text)    
    else:
        dataset = ir_datasets.load(cfg.input_path)
        if cfg.type == "query":
            for doc in tqdm(dataset.queries_iter(), desc=f"Reading data from ir_datasets {cfg.input_path}"):
                idx = doc.query_id
                text = doc.text.strip()
                text = prompt + text
                ids.append(idx)
                texts.append(text)
        else:
            for doc in tqdm(dataset.docs_iter(), desc=f"Reading data from ir_datasets {cfg.input_path}"):
                idx = doc.doc_id
                try:
                    text = (doc.title + " " + doc.text).strip()
                except:
                    text = (doc.text).strip()
                text = prompt + text
                ids.append(idx)
                texts.append(text)
    assert len(ids) == len(texts)
    # tokens in the vocabulary
    all_token_ids = list(range(tokenizer.get_vocab_size()))
    all_tokens = np.array(tokenizer.convert_ids_to_tokens(all_token_ids))
    prompt_token_ids, prompt_len = get_token_ids_in_text(prompt, tokenizer, nltk_tokenize=True)

    for idx in tqdm(range(0, len(ids), cfg.batch_size)):
        logger.log(msg={"batch": idx},level=1)
        batch_texts = texts[idx : idx + cfg.batch_size]
        batch_ids = ids[idx : idx + cfg.batch_size]
        batch_tkn = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.input_max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        ).to(device)
        if cfg.fp16:
            with torch.no_grad(), torch.cuda.amp.autocast():
                if cfg.type == "query":
                    batch_output = model.encode_queries(**batch_tkn).to("cpu")
                else:
                    batch_output = model.encode_docs(**batch_tkn).to("cpu")
        else:
            with torch.no_grad():
                if cfg.type == "query":
                    batch_output = model.encode_queries(**batch_tkn).to("cpu")
                else:
                    batch_output = model.encode_docs(**batch_tkn).to("cpu")
        batch_output = batch_output.float()
        batch_output = batch_output[:, :50265] # (bs, seq, vs)

        all_token_ids = list(range(tokenizer.get_vocab_size()))
        batch_indices_in_all_token_ids_in_text = []
        for text in batch_texts:
            # words = [i for i in word_tokenize(text.lower()) if i.lower() not in stopwords.words('english')]
            # token_ids_in_text = tokenizer(" ".join(words)).input_ids
            token_ids_in_text, length = get_token_ids_in_text(text, tokenizer, nltk_tokenize=True)
            token_ids_in_text = token_ids_in_text[prompt_len:]

            indices_in_all_token_ids_in_text = [all_token_ids.index(token_id) for token_id in token_ids_in_text if token_id in all_token_ids]
            batch_indices_in_all_token_ids_in_text.append(indices_in_all_token_ids_in_text)

        if cfg.in_text_only:
            top_k_res = batch_output.topk(dim=1, k=cfg.top_k, sorted=False)
            batch_topk_values = (top_k_res.values * cfg.scale_factor).to(torch.int)
            topk_indices = top_k_res.indices
            batch_topk_tokens = all_tokens[topk_indices]

            # do non-zero selection
            batch_output = (batch_output * cfg.scale_factor).to(torch.int)
            batch_tokens = []
            batch_token_ids = []
            batch_weights = []
                        
            for indices_in_all_token_ids_in_text, output in zip(batch_indices_in_all_token_ids_in_text, batch_output.tolist()):
                batch_tokens.append([all_tokens[indices] for indices in indices_in_all_token_ids_in_text])
                batch_token_ids.append([all_token_ids[indices] for indices in indices_in_all_token_ids_in_text])
                batch_weights.append([output[indices] for indices in indices_in_all_token_ids_in_text])

            for text_id, text, tokens, weights , token_ids, topk_tokens, topk_values  in zip(
                batch_ids, batch_texts, batch_tokens, batch_weights, batch_token_ids, batch_topk_tokens, batch_topk_values
            ):
                if all(v == 0 for v in weights):
                    result = {"id": text_id, "text": text, "tokens": topk_tokens, "weights": topk_values, "vector": dict(zip(topk_tokens, topk_values))}
                else:
                    result = {"id": text_id, "text": text, "tokens": tokens, "weights": weights, "vector": dict(zip(tokens, weights))}
                # print({"text": text, "tokens": tokens, "weights": weights})

                write_to_file(
                    file_writer,
                    result,
                    cfg.type,
                )
if __name__ == "__main__":
    start_time = time.time()
    inference()
    run_time = time.time() - start_time
    logger.log(msg=f"Finished! Runing time {str(datetime.timedelta(seconds=666))}",level=1)
