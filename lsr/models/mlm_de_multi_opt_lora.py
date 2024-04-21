from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch
from torch import nn
from transformers import PretrainedConfig,AutoTokenizer,AutoModelForSeq2SeqLM,AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType,AdaLoraConfig,AutoPeftModelForSeq2SeqLM,PeftModel,AutoPeftModelForCausalLM

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


class TransformerMLMOPTLoraDecoderMultiStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerMLMSparseEncoder
    """

    model_type = "MLM_OPT_LORA_DECODER_MULTISTEPS"

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
        self.lora_pretrianed = False
        super().__init__(**kwargs)


class TransformerMLMSparseOPTLoraDecoderMultiSteps(SparseEncoder):
    """
    TransformerSeq2SeqSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLMOPTLoraDecoderMultiStepsConfig

    def __init__(self, 
                 config: TransformerMLMOPTLoraDecoderMultiStepsConfig = TransformerMLMOPTLoraDecoderMultiStepsConfig(),
                 model=None):
        
        super(SparseEncoder, self).__init__(config)
        if config.lora_pretrianed:
            print("loading pretrained lora model", config)
            self.model = self.load_peft_model(config._name_or_path)
        else: 
            print("build lora model", config)
            self.model = self.build_peft_model(config.tf_base_model_name_or_dir)

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

    def build_peft_model(self, model_name_or_dir):
        model = AutoModelForCausalLM.from_pretrained(model_name_or_dir,trust_remote_code=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules =["down_proj", "gate_proj", "q_proj", "o_proj", "up_proj", "v_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        return model

    def load_peft_model(self, model_name_or_dir):
        model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_dir)
        return model