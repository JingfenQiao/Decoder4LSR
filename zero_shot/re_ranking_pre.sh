CUDA_VISIBLE_DEVICES=0 python query_expansion.py \
    --queries_path /var/scratch/jkang/project/lsr_eval/data/msmarco/TREC_DL_2019/queries_2019/raw.tsv \
    --collections_path /var/scratch/jkang/project/lsr_eval/data/msmarco/full_collection/raw.tsv \
    --query_output_path  /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/dl19_query_expansion.json \
    > dl19_query_expansion.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python input_expansion.py \
    --query_input_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/dl19_query_expansion.json \
    --query_output_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/dl19_query_expansion.tsv \
    > dl19_query_expansion.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python re_ranking_preprocess.py \
    --queries_path /var/scratch/jkang/project/lsr_eval/data/msmarco/TREC_DL_2019/queries_2019/raw.tsv \
    --collections_path /var/scratch/jkang/project/lsr_eval/data/msmarco/full_collection/raw.tsv \
    --passage_output_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_passage_test.json \
    --query_output_path  /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_passage_test.json \
    > re_ranking_preprocess_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python re_ranking.py \
    --query_input_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_query.json \
    --passage_input_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_passage.json \
    --output_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/reranking.trec \
    > reranking.log 2>&1 &