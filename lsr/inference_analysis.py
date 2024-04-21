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

logger = logging.getLogger(__name__)


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
    # Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    # output_path = Path(cfg.output_dir).joinpath(cfg.output_file)
    # file_writer = open(output_path, "w")

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


    all_token_ids = list(range(tokenizer.get_vocab_size()))
    all_tokens = np.array(tokenizer.convert_ids_to_tokens(all_token_ids))

    batch_texts = ["Query: is there any other visilization methods?"]
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
    batch_output = batch_output[:, :50265]
    # print(batch_output)
    print(batch_output.shape)
    non_zero_indices = torch.nonzero(batch_output, as_tuple=False)
    non_zero_values = batch_output[non_zero_indices[:,0], non_zero_indices[:,1]]
    print(non_zero_values)
    print(non_zero_indices)
    
    return batch_output



        # if cfg.top_k > 0:
        #     # do top_k selection in batch
        #     top_k_res = batch_output.topk(dim=1, k=cfg.top_k, sorted=False)
        #     batch_values = (top_k_res.values * cfg.scale_factor).to(torch.int)
        #     indices = top_k_res.indices
        #     batch_tokens = all_tokens[indices]
        #     for text_id, text, tokens, weights in zip(
        #         batch_ids, batch_texts, batch_tokens, batch_values
        #     ):
        #         mask = weights > 0
        #         tokens = tokens[mask]
        #         weights = weights[mask]
        #         write_to_file(
        #             file_writer,
        #             {
        #                 "id": text_id,
        #                 "text": text,
        #                 "vector": dict(zip(tokens.tolist(), weights.tolist())),
        #             },
        #             cfg.type,
        #         )
        # else:
        #     # do non-zero selection
        #     batch_output = (batch_output * cfg.scale_factor).to(torch.int)
        #     batch_tokens = [[] for _ in range(len(batch_ids))]
        #     batch_weights = [[] for _ in range(len(batch_ids))]
        #     for row_col in batch_output.nonzero():
        #         row, col = row_col
        #         batch_tokens[row].append(all_tokens[col].item())
        #         batch_weights[row].append(batch_output[row, col].item())
        #     for text_id, text, tokens, weights in zip(
        #         batch_ids, batch_texts, batch_tokens, batch_weights
        #     ):
        #         write_to_file(
        #             file_writer,
        #             {"id": text_id, "text": text, "vector": dict(zip(tokens, weights))},
        #             cfg.type,
        #         )


if __name__ == "__main__":
    start_time = time.time()
    inference()

    run_time = time.time() - start_time
    logger.log(msg=f"Finished! Runing time {str(datetime.timedelta(seconds=666))}",level=1)
