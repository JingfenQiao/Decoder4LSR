
CUDA_VISIBLE_DEVICES=0  nohup python -m scripts/try.py


CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=splade_asm_msmarco_distil_flops_0.0_0.5 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_only_distil_t5_base training_arguments.fp16=True wandb.resume=False \
> log/lsr_test.log 2>&1 &

# MLM
CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_only_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_decoder_only_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_decoder_multi_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_decoder_single_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

# MLP
CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_encoder_only_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_encoder_only_t5_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_decoder_only_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_decoder_only_t5_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_encoder_decoder_multi_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_encoder_decoder_multi_t5_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_encoder_decoder_single_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_encoder_decoder_single_t5_base.log 2>&1 &

# t5_base mlp&mlm
CUDA_VISIBLE_DEVICES=0 nohup python -m lsr.train +experiment=qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08 training_arguments.fp16=True > log/qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08.log 2>&1 &

# t5_base binary&mlm delete
CUDA_VISIBLE_DEVICES=0 nohup python -m lsr.train +experiment=qbin_dmlm_encoder_decoder_multi_t5_base_0.0_0.08 training_arguments.fp16=True > log/qbin_dmlm_encoder_decoder_multi_t5_base_0.0_0.08.log 2>&1 &

# t5_base mlm&mlp
CUDA_VISIBLE_DEVICES=0 nohup python -m lsr.train +experiment=qmlm_dmlp_encoder_decoder_multi_t5_base_0.1_0.0 training_arguments.fp16=True > log/qmlm_dmlp_encoder_decoder_multi_t5_base_0.1_0.0.log 2>&1 &

# OPT3.5 decoder-only MLP
CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_decoder_only_opt3.5 training_arguments.fp16=True wandb.resume=False > log/mlp_decoder_only_opt3.5.log 2>&1 &



# OPT27 decoder-only MLP
CUDA_VISIBLE_DEVICES=0,1 nohup python -m lsr.train +experiment=mlm_decoder_only_opt27_lora_0.001 training_arguments.fp16=True wandb.resume=False > log/mlm_decoder_only_opt27_0.001.log 2>&1 &

