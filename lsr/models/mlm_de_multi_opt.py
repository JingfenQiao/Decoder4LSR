from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch
from torch import nn
from transformers import PretrainedConfig,AutoTokenizer,AutoModelForSeq2SeqLM,AutoModelForCausalLM
import torch

class TransformerMLMOPTDecoderMultiStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerMLMSparseEncoder
    """

    model_type = "MLM_OPT_DECODER_MULTISTEPS"

    def __init__(
        self,
        tf_base_model_name_or_dir: str = "facebook/opt-350m",
        pool: str = "max",
        activation: str = "relu",
        norm: str = "log1p",
        **kwargs,
    ):
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.pool = pool
        self.activation = activation
        self.norm = norm
        super().__init__(**kwargs)


class TransformerMLMSparseOPTDecoderMultiSteps(SparseEncoder):
    """
    TransformerSeq2SeqSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLMOPTDecoderMultiStepsConfig

    def __init__(self, 
                 config: TransformerMLMOPTDecoderMultiStepsConfig = TransformerMLMOPTDecoderMultiStepsConfig(),
                 model=None):
        
        super(SparseEncoder, self).__init__(config)
        print("model name ", config.tf_base_model_name_or_dir)
        self.model = AutoModelForCausalLM.from_pretrained(config.tf_base_model_name_or_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tf_base_model_name_or_dir)

        self.activation = FunctionalFactory.get(config.activation)
        self.pool = PoolingFactory.get(config.pool)

        self.norm = FunctionalFactory.get(config.norm)

    def forward(self, **kwargs):
        special_tokens_mask = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs,output_hidden_states=True)

        logits = (
            output.logits
            * kwargs["attention_mask"].unsqueeze(-1)
            * (1 - special_tokens_mask).unsqueeze(-1)
        )
        logits = self.norm(self.activation(logits))
        lex_weights = self.pool(logits)
        return SparseRep(dense=lex_weights)