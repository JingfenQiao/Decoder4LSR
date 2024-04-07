## CREAT TOP100 PASSAGE COLLECTIONS

CUDA_VISIBLE_DEVICES=0 python re_ranking.py \
    --query_input_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_query.json \
    --passage_input_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_passage.json \
    --qrels /var/scratch/jkang/project/lsr_eval/data/msmarco/TREC_DL_2019/qrel.json \
    --output_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/dl19_reranking_run.trec > test.log 2>&1 &

## CREAT QUERY SPARSE REPRESENTAION


## CREAT PASSAGE SPARSE REPRESENTAION


## BUILD INDEX


## SEARCH 


## EVALUATION