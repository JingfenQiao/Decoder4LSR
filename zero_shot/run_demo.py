from nltk.stem import PorterStemmer
import json
docs = []
ps = PorterStemmer()
with open("data/msmarco-passage/gpt_bow_psgs.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        stemed_vector = {}
        for word in doc["vector"]:
            try:
                doc["vector"][word] = float(doc["vector"][word])
            except:
                continue
            for tok in word.split():
                tok = ps.stem(tok)
                if tok in stemed_vector:
                    stemed_vector[tok] = max(
                        stemed_vector[tok], doc["vector"][word])
                else:
                    stemed_vector[tok] = doc["vector"][word]
        doc["vector"] = stemed_vector
        docs.append(doc)

with open("data/msmarco-passage/gpt_bow_queries.jsonl", "r") as f, open("run_dl19_gpt3.5.trec", "w") as frun:
    for line in f:
        query = json.loads(line.strip())
        stemed_vector = {}
        for word in query["vector"]:
            try:
                query["vector"][word] = float(query["vector"][word])
            except:
                continue
            for tok in word.split():
                tok = ps.stem(tok)
                if tok in stemed_vector:
                    stemed_vector[tok] = max(
                        stemed_vector[tok], float(query["vector"][word]))
                else:
                    stemed_vector[tok] = float(query["vector"][word])
        query["vector"] = stemed_vector
        dscores = []
        for doc in docs:
            try:
                score = 0.0
                for word in query["vector"]:
                    if word in doc["vector"]:
                        score += float(query["vector"][word]
                                       ) * float(doc["vector"][word])
                dscores.append((score, doc["id"], doc["text"]))
            except:
                print(query)
                print(doc)

        dscores = sorted(dscores, reverse=True)
        for rank, row in enumerate(dscores[:1000]):
            frun.write(
                f"{query['id']}\tQ0\t{row[1]}\t{rank}\t{row[0]}\tgpt3.5-1-shot\n")
        # print(f"Query {query['text']}")
        # for score, did, dtext in dscores[:10]:
        #     print(f"Score: {score} DID: {did} Text: {dtext}\n")
        #     print("=================\n")
        # input()
