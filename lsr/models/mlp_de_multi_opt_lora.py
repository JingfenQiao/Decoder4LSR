from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch
from torch import nn
from transformers import PretrainedConfig,AutoTokenizer,AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType,AdaLoraConfig,AutoPeftModelForSeq2SeqLM,PeftModel
import logging 

logger = logging.getLogger(__name__)
class TransformerMLPOPTDecoderMultiStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerMLPSparseEncoder
    """

    model_type = "MLP_OPT_DECODER_MULTISTEPS"

    def __init__(
        self,
        tf_base_model_name_or_dir: str = "",
        activation: str = "relu",
        norm: str = "log1p",
        scale=1.0,
        **kwargs,
    ):
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.activation = activation
        self.norm = norm
        self.scale = scale
        super().__init__(**kwargs)

class TransformerMLPSparseOPTDecoderMultiSteps(SparseEncoder):
    """
    TransformerSeq2SeqSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLPOPTDecoderMultiStepsConfig

    def __init__(self, 
                 config: TransformerMLPOPTDecoderMultiStepsConfig = TransformerMLPOPTDecoderMultiStepsConfig()):        
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(config.tf_base_model_name_or_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        # self.linear = nn.Linear(512, 1)
        self.activation = FunctionalFactory.get(config.activation)
        self.norm = FunctionalFactory.get(config.norm)
        self.scale = nn.Parameter(torch.tensor(config.scale))

    def forward(self, to_scale=False, **kwargs):
        special_tokens_mask = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs,output_hidden_states=True)
        decoder_last_hidden_state = output.hidden_states[-1] #(batch_size, sequence_length, hidden_size)     
        tok_weights = self.linear(decoder_last_hidden_state).squeeze(-1)  # bs x len x 1

        tok_weights = (
            self.norm(self.activation(tok_weights))
            * kwargs["attention_mask"]
        )

        if to_scale:
            tok_weights = tok_weights * self.scale
        size = torch.tensor(
            (tok_weights.size(0), self.model.config.vocab_size),
            device=tok_weights.device,
        )
        return SparseRep(indices=kwargs["input_ids"], values=tok_weights, size=size)