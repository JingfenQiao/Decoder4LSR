# Imports
import logging
import json
import argparse
from transformers import AutoTokenizer
import tqdm

def main(infile_path: str, outfile_path: str, tokenizer: AutoTokenizer):
    """ Reads data from a file, processes it, and writes to an output file. """
    with open(infile_path, "r") as infn, open(outfile_path, "w") as outfn:
        for line in tqdm.tqdm(infn):
            doc = json.loads(line)
            tokens = list(doc["vector"].keys())
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            index_doc_id = dict(zip(token_ids, doc["vector"].values()))
            doc["vector"] = index_doc_id
            doc.pop("text")
            outfn.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument('model_name', type=str, help='Name of the model for the tokenizer.')
    parser.add_argument('infile_path', type=str, help='Path to the input file.')
    parser.add_argument('outfile_path', type=str, help='Path to the output file.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    main(args.infile_path, args.outfile_path, tokenizer)