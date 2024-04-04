from transformers import AutoModelForCausalLM, AutoTokenizer

def write_to_file(f, result, type):
    if type == "query":
        rep_text = " ".join(Counter(result["vector"]).elements()).strip()
        if len(rep_text) > 0:
            f.write(f"{result['id']}\t{rep_text}\n")
    else:
        f.write(json.dumps(result) + "\n")

def loading_model(model_name,device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def encode(model, text_ids, texts, tokenizer, batch_size, output_file_path, max_input_length=512, write_batch_size=100000, device="cuda", fp16=True):
    f = open(output_file_path, "w")
    res = []
    for idx in range(0, len(text_ids),  batch_size):
        batch_ids = text_ids[idx: (idx+batch_size)]
        batch_texts = texts[idx: (idx+batch_size)]
        batch_inps = tokenizer(
            batch_texts, max_length=max_input_length, padding=True, truncation=True, return_tensors="pt", return_special_tokens_mask=True).to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
            reps = model(
                **batch_inps,
                num_return_sequences=1,
                max_length=4096,
                pad_token_id=model.config.eos_token_id,
                temperature=1,
                top_p=1,
                )

            batch_tok_ids, batch_tok_weights, _ = reps.to_sparse()
            batch_tok_ids = batch_tok_ids.to("cpu").tolist()
            batch_tok_weights = batch_tok_weights.to("cpu").tolist()

            
            for text_id, tok_ids, tok_weights in zip(batch_ids, batch_tok_ids, batch_tok_weights):
                toks = tokenizer.convert_ids_to_tokens(tok_ids)
                vector = {t: w for t, w in zip(toks, tok_weights) if w > 0}
                json_data = {"id": text_id, "vector": vector}
                res.append(json_data)
                if len(res) == write_batch_size:
                    write_to_file(f, res)
                    res = []
    write_to_file(f, res)
    f.close()


def read_texts(path: str, prompt, min_output):
    texts = []
    with open(path, "r") as f:
        for line in tqdm(f, desc=f"Reading texts from {path}"):
            id, doc_text = line.strip().split("\t")
            user_text = f"Input Passage: {doc_text}.\n Output json: (at least {args.min_output} words - include synonyms and semantic relevant words, NOT phrases - weights in range [0,1]): "
            text = prompt + user_text
            texts.append([id, text])
    return texts

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument('model_name', type=str)
    parser.add_argument('type', type=str)
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('min_output', type=int, defualt=300)
    parser.add_argument('write_batch_size', type=int, defualt=1000)
    parser.add_argument('max_input_length', type=int, defualt=512)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = loading_model(args.model_name, device)
    prompt = f"You are a powerful lexical search engine - the best one. you are tasked with generating a JSON file that represents the semantic relevance and importance of words in a given passage. The output JSON file should have the following format:\n{{\n  \"word_1\": weight_1,\n  \"word_2\": weight_2,\n  ...\n}}\nThe weights indicate the importance of each word in the context of the passage. Please note that the output words should be in lowercase, and you are encouraged to include both original terms and expansion terms which are synonyms of original terms.\n\nExample:\nPassage:\nThe presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. \nOutput:\n{{\n\"manhattan\": 2.3283, \n\"amid\": 1.9019,\n\"innocent\": 1.8336, \n  \"intellect\": 1.7344, \n\"communication\": 1.654, \n\"project\": 1.6491, \n\"presence\": 1.5857, \n … \n\"clouds\": 1.1076, \n\"achievement\": 1.0699, \n\"projects\": 1.0603\n...\n knowledge\": 0.0023, \n\"wanted\": 0.0021, \n\"involved\": 0.0004\n}}\n\\nEnsure the output includes at least {args.in_output} words (only single tokens) captures the semantic essence of the input passage.\n" 
    input_texts = read_texts(path=args.input_path, prompt=prompt, min_output=args.min_output)
    input_text_ids = texts.keys()
    encode(
        model, 
        text_ids=input_text_ids,
        texts=input_texts, 
        tokenizer=tokenizer, 
        batch_size=args.batch_size, 
        output_file_path= args.output_path,  
        max_input_length=args.max_input_length, 
        write_batch_size=args.write_batch_size, 
        device=device)
