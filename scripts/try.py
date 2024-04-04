from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

doc_text="LLMs are artificial neural networks. The largest and most capable, as of March 2024, are built with a decoder-only transformer-based architecture while some recent implementations are based on other architectures, such as recurrent neural network variants and Mamba (a state space model)."
min_output=300
prompt = f"You are a powerful lexical search engine - the best one. you are tasked with generating a JSON file that represents the semantic relevance and importance of words in a given passage. The output JSON file should have the following format:\n{{\n  \"word_1\": weight_1,\n  \"word_2\": weight_2,\n  ...\n}}\nThe weights indicate the importance of each word in the context of the passage. Please note that the output words should be in lowercase, and you are encouraged to include both original terms and expansion terms which are synonyms of original terms.\n\nExample:\nPassage:\nThe presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. \nOutput:\n{{\n\"manhattan\": 2.3283, \n\"amid\": 1.9019,\n\"innocent\": 1.8336, \n  \"intellect\": 1.7344, \n\"communication\": 1.654, \n\"project\": 1.6491, \n\"presence\": 1.5857, \n … \n\"clouds\": 1.1076, \n\"achievement\": 1.0699, \n\"projects\": 1.0603\n...\n knowledge\": 0.0023, \n\"wanted\": 0.0021, \n\"involved\": 0.0004\n}}\n\\nEnsure the output includes at least {min_output} words (only single tokens) captures the semantic essence of the input passage.\n"
user_text = f"Input Passage: {doc_text}.\n Output json: (at least {min_output} words - include synonyms and semantic relevant words, NOT phrases - weights in range [0,1]): "

input_ids = tokenizer(prompt + user_text, return_tensors="pt")
start = time.time()
output = model.generate(
    **input_ids, 
    num_return_sequences=1,
    max_length=4096,
    pad_token_id=model.config.eos_token_id,
    temperature=1,
    top_p=1,
    )
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
end = time.time()
print(end - start)
print(generated_text)
