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

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class TransformerMLPT5DecoderMultiStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerMLPSparseEncoder
    """

    model_type = "MLP_T5_DECODER_MULTISTEPS"

    def __init__(
        self,
        tf_base_model_name_or_dir: str = "google/flan-t5-base",
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

class TransformerMLPSparseT5DecoderMultiSteps(SparseEncoder):
    """
    TransformerSeq2SeqSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLPT5DecoderMultiStepsConfig

    def __init__(self, 
                 config: TransformerMLPT5DecoderMultiStepsConfig = TransformerMLPT5DecoderMultiStepsConfig()):        
        super().__init__(config)
        self.model = self.build_model(config.tf_base_model_name_or_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tf_base_model_name_or_dir
        )
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.activation = FunctionalFactory.get(config.activation)
        self.norm = FunctionalFactory.get(config.norm)
        self.scale = nn.Parameter(torch.tensor(config.scale))

    def forward(self, to_scale=False, **kwargs):
        if "special_tokens_mask" in kwargs:
            kwargs.pop("special_tokens_mask")

        kwargs["input_ids"] = self.model._shift_right(kwargs["input_ids"])
        decoder_outputs = self.model.decoder(**kwargs,output_hidden_states=True)
        decoder_last_hidden_state = decoder_outputs.last_hidden_state #(batch_size, sequence_length, hidden_size)
        # decoder_last_hidden_state = decoder_outputs.decoder_hidden_states[-1]
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


    def build_model(self, model_name_or_dir):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_dir)
        return model