
# Leveraging Decoder Architectures for Learned Sparse Retrieval

The structure of the `lsr` package is as following: 

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

* The list of all configurations used in the paper could be found [here](#list-of-configurations-used-in-the-paper)

* The instruction for running experiments could be found [here](#training-and-inference-instructions)

## Training and inference instructions 

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

### 2. Downwload/Prepare datasets
 We have included all pre-defined dataset configurations under `lsr/configs/dataset`. Before starting training, ensure that you have the `ir_datasets` and (huggingface) `datasets` libraries installed, as the framework will automatically download and store the necessary data to the correct directories.

For datasets from `ir_datasets`, the downloaded files are saved by default at `~/.ir_datasets/`. You can modify this path by changing the `IR_DATASETS_HOME` environment variable.

Similarly, for datasets from the HuggingFace's `datasets`, the downloaded files are stored at `~/.cache/huggingface/datasets` by default. To specify a different cache directory, set the `HF_DATASETS_CACHE` environment variable. 

To train a customed model on your own dataset, please use the sample configurations under `lsr/config/dataset` as templates. Overall, you need three important files (see `lsr/dataset_utils` for the file format): 
- document collection: maps `document_id` to `document_text` 
- queries: maps `query_id` to `query_text`
- train triplets or scored pairs:
    - train triplets, used for contrastive learning, contains a list of <`query_id`, `positive_document_id`, `negative_document_id`> triplets.
    - scored_pairs, used for distillation training, contain pairs of <`query`, `document_id`> with a relevance score.  