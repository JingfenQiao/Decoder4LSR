
# Leveraging Decoder Architectures for Learned Sparse Retrieval

This repository contains the code used to reproduce the training of the paper in [*Leveraging Decoder Architectures for Learned Sparse Retrieval*]() paper. 


# Introduction

Learned Sparse Retrieval (LSR) has traditionally focused on small-scale encoder-only transformer architectures. With the advent of large-scale pre-trained language models, their capability to generate sparse representations for retrieval tasks across different transformer-based architectures, including encoder-only, decoder-only, and encoder-decoder models, remains largely unexplored. This repository investigates the effectiveness of LSR across these architectures, exploring various sparse representation heads and model scales.

Our code ultilize the [*Unified LSR Framework*](https://github.com/thongnt99/learned-sparse-retrieval),  the structure of the `lsr` package is as following: 

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
Download Ms Marco training data from [*here*](https://download.europe.naverlabs.com/splade/sigir22/data.tar.gz). After downloading, you can just untar in the root directory, and it will be placed in the right folder.

```
tar -xzvf file.tar.gz

mkdir -p data/msmarco/full_collection/split

# Split the file into 5 parts and save in the split folder
split -n 5 -a 2 raw.tsv splits/raw_split_ 

```

# Index and Search

You can download the [*Anserini*](https://github.com/thongnt99/anserini-lsr) for index and search, then follow the instructions in the README for installation. If the tests fail, you can skip it by adding -Dmaven.test.skip=true. When the installation is done, you can continue with the next steps.


## List of configurations used in the paper

- RQ1: Can LLMs effectively generate sparse representations in a zero-shot setting when prompted?

Results of RQ1 in Table 1 are the outputs of the experiments run using scripts in bash_rq1.

- RQ2: Can encoder-decoder or decoder-only backbones outperform encoder-only backbones when using multi-tokens decoding approach?


Results of RQ2 in Table 2 are the outputs of the experiments run using scripts in bash_rq2.

- RQ3: Which sparse representation head is better for creating a sparse representation?

Results of RQ3 in Table 3 are the outputs of the experiments run using scripts in bash_rq3.


- RQ4: How is the performance of the LSR affected by scaling the teacher and student models on the different backbones?

Results of RQ4 in Table 4 are the outputs of the experiments run using scripts in bash_rq4.
