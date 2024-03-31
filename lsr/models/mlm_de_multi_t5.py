from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch
from torch import nn
from transformers import PretrainedConfig,AutoTokenizer,AutoModelForSeq2SeqLM
import torch
from peft import LoraConfig, get_peft_model, TaskType,AdaLoraConfig,AutoPeftModelForSeq2SeqLM,PeftModel

class TransformerMLMT5DecoderMultiStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerMLMSparseEncoder
    """

    model_type = "MLM_T5_DECODER_MULTISTEPS"

    def __init__(
        self,
        tf_base_model_name_or_dir: str = "google/flan-t5-base",
        pool: str = "max",
        activation: str = "relu",
        norm: str = "log1p",
        term_importance: str = "no",
        doc_quality: str = "no",
        **kwargs,
    ):
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.pool = pool
        self.activation = activation
        self.norm = norm
        super().__init__(**kwargs)


class TransformerMLMSparseT5DecoderMultiSteps(SparseEncoder):
    """
    TransformerSeq2SeqSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLMT5DecoderMultiStepsConfig

    def __init__(self, 
                 config: TransformerMLMT5DecoderMultiStepsConfig = TransformerMLMT5DecoderMultiStepsConfig(),
                 model=None):
        
        super(SparseEncoder, self).__init__(config)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.tf_base_model_name_or_dir)
        self.activation = FunctionalFactory.get(config.activation)
        self.pool = PoolingFactory.get(config.pool)
        self.norm = FunctionalFactory.get(config.norm)

    def forward(self, **kwargs):
        if "special_tokens_mask" in kwargs:
            kwargs.pop("special_tokens_mask")

        kwargs["input_ids"] = self.model._shift_right(kwargs["input_ids"])
        decoder_outputs = self.model.decoder(**kwargs,output_hidden_states=True)
        
        # get last_hidden_states
        last_hidden_state = decoder_outputs.last_hidden_state #(batch_size, sequence_length, hidden_size)
        logits = self.model.lm_head(last_hidden_state)
        shifted_attention_mask = self.model._shift_right(kwargs["attention_mask"])

        logits = (
            logits
            * shifted_attention_mask.unsqueeze(-1)
        )
        # norm default: log(1+x)
        logits = self.norm(self.activation(logits))
        lex_weights = self.pool(logits)
        return SparseRep(dense=lex_weights)