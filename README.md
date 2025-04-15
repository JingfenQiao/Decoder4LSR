
# Leveraging Decoder Architectures for Learned Sparse Retrieval

This repository contains the code used to reproduce the training of the paper in [*Leveraging Decoder Architectures for Learned Sparse Retrieval*]() paper. 


# Introduction

Learned Sparse Retrieval (LSR) has traditionally focused on small-scale encoder-only transformer architectures. With the advent of large-scale pre-trained language models, their capability to generate sparse representations for retrieval tasks across different transformer-based architectures, including encoder-only, decoder-only, and encoder-decoder models, remains largely unexplored. This repository investigates the effectiveness of LSR across these architectures, exploring various sparse representation heads and model scales.

Our code ultilize the [*Unified LSR Framework*](https://github.com/thongnt99/learned-sparse-retrieval) code,  the structure of the `lsr` package is as following: 

```.
├── configs  #configuration of different components
│   ├── dataset 
│   ├── experiment #define exp details: dataset, loss, model, hp 
│   ├── loss 
│   ├── model
│   └── wandb
├── datasets    #implementations of dataset loading & collator
├── losses  #implementations of different losses + regularizer
├── models  #implementations of different models
├── tokenizer   #a wrapper of HF's tokenizers
├── trainer     #trainer for training 
└── utils   #common utilities used in different places
```

## Setup

### 1. Create conda environment and install dependencies: 

Create `conda` environemt:
```
conda create --name lsr python=3.9.12
conda activate lsr
```
Install dependencies with `pip`
```
pip install -r requirements.txt
```

## Dataset and Models


## List of configurations used in the paper

- RQ1: Can LLMs effectively generate sparse representations in a zero-shot setting when prompted?

Results of RQ1 in Table 1 are the outputs of the experiments run using scripts in bash_rq1.

- RQ2: Can encoder-decoder or decoder-only backbones outperform encoder-only backbones when using multi-tokens decoding approach?


Results of RQ2 in Table 2 are the outputs of the experiments run using scripts in bash_rq2.

- RQ3: Which sparse representation head is better for creating a sparse representation?

Results of RQ3 in Table 3 are the outputs of the experiments run using scripts in bash_rq3.


- RQ4: How is the performance of the LSR affected by scaling the teacher and student models on the different backbones?

Results of RQ3 in Table 3 are the outputs of the experiments run using scripts in bash_rq3.
