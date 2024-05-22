import ir_measures
from ir_measures import *
from scipy import stats
import sys
import ir_datasets

run1_path = sys.argv[1]
run2_path = sys.argv[2]
qrel_path = sys.argv[3]

def process_file(run_file):
    run = {}
    dataset = None
    if 'dl19' in run_file:
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
    elif 'dl20' in run_file:
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2020/judged")
    
    if dataset is None:
        raise ValueError("Dataset could not be loaded. Please check the run file name and corresponding dataset.")

    dataset_ids = {query.query_id for query in dataset.queries_iter()}
    
    with open(run_file, "r") as infn:
        for line in infn:
            parts = line.strip().split()
            qid, docid, rank, score = parts[0], parts[2], parts[3], parts[4]
            if qid not in dataset_ids:
                continue
            if qid not in run:
                run[qid] = {}
            if int(rank) <= 10:
                run[qid][docid] = float(score)
    return run

qrels = list(ir_measures.read_trec_qrels(qrel_path))

def load_run(run_path):
    if "msmarco" in run_path:
        return ir_measures.read_trec_run(run_path)
    else:
        return process_file(run_path)

run1 = load_run(run1_path)
run2 = load_run(run2_path)

scores1 = {metric.query_id: metric.value for metric in ir_measures.iter_calc([Recall @ 1000], qrels, run1)}
scores2 = {metric.query_id: metric.value for metric in ir_measures.iter_calc([Recall @ 1000], qrels, run2)}

l1 = [scores1[qid] for qid in scores1]
l2 = [scores2[qid] for qid in scores1 if qid in scores2]

result = stats.ttest_rel(l1, l2)
print(result)
