from datasets import load_dataset
dataset = load_dataset('intfloat/query2doc_msmarco')

with open("/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/TREC_DL_2019/queries_2019/query2doc_lower.tsv", "w") as outfn:
    for data in dataset['trec_dl2019']:
        query_id, query, pseudo_doc = data["query_id"], data["query"], data["pseudo_doc"]
        text = query + " " + pseudo_doc.lower()
        outfn.write(f"{query_id}\t{text}\n")


with open("/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/TREC_DL_2019/queries_2019/query2doc_5times_query_lower.tsv", "w") as outfn:
    for data in dataset['trec_dl2019']:
        query_id, query, pseudo_doc = data["query_id"], data["query"], data["pseudo_doc"]
        text = query + " " + query +  " " + query + " " + query + " " + query + " " + pseudo_doc.lower()
        outfn.write(f"{query_id}\t{text}\n")