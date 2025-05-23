from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch
from torch import nn
from transformers import PretrainedConfig,AutoTokenizer,AutoModelForSeq2SeqLM
import torch
# from peft import LoraConfig, get_peft_model, TaskType,AdaLoraConfig,AutoPeftModelForSeq2SeqLM,PeftModel

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

class EPICTermImportance(nn.Module):
    """
    EPICTermImportance class
    This module is used in EPIC model to estimate the term importance for each input term
    Paper: https://arxiv.org/abs/2004.14245
    """

    def __init__(self, dim: int = 768) -> None:
        """
        Construct an EPICTermImportance module
        Parameters
        ----------
        dim: int
            dimension of the input vectors
        """
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        # inputs usually has shape: batch_size x seq_length x vector_size
        s = torch.log1p(self.softplus(self.linear(inputs)))
        return s


class EPICDocQuality(nn.Module):
    """
    EpicDocQuality
    This module is used in EPIC model to estimate the doc quality's score. The input vector is usually the [CLS]'s embedding.
    Paper: https://arxiv.org/abs/2004.14245
    """

    def __init__(self, dim: int = 768) -> None:
        """
        Construct an EpicDocquality module
        Parameters
        ----------
        dim: int
            dimension of the input vector
        """
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.sm = nn.Sigmoid()

    def forward(self, inputs):
        """forward function"""
        s = self.sm(self.linear(inputs))
        return s

class TransformerMLMEncoderDecoderMultiStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerMLMSparseEncoder
    """

    model_type = "MLM_EncoderDecoder_MULTISTEPS"

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
        """
        Construct an instance of TransformerSeq2SeqConfig
        Paramters
        ---------
        tf_base_model_name_or_dir: str
            name/path of the pretrained weights (HuggingFace) for initializing the masked language model's backbone
        pool: str
            pooling strategy (max, sum)
        activation: str
            activation function
        norm: str
            weight normalization function
        term_importance: str
            module for estimating term importance. "no" for ormitting this component
        doc_quality: str
        """
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.pool = pool
        self.activation = activation
        self.norm = norm
        self.term_importance = term_importance
        self.doc_quality = doc_quality
        super().__init__(**kwargs)


class TransformerMLMSparseEncoderDecoderMultiSteps(SparseEncoder):
    """
    TransformerSeq2SeqSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLMEncoderDecoderMultiStepsConfig

    def __init__(self, 
                 config: TransformerMLMEncoderDecoderMultiStepsConfig = TransformerMLMEncoderDecoderMultiStepsConfig(),
                 model=None):
        
        super(SparseEncoder, self).__init__(config)
        self.model = self.build_model(config.tf_base_model_name_or_dir)
    
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tf_base_model_name_or_dir
        )
        self.activation = FunctionalFactory.get(config.activation)
        self.pool = PoolingFactory.get(config.pool)
        if config.term_importance == "no":
            self.term_importance = functional.AllOne()
        elif config.term_importance == "epic":
            self.term_importance = EPICTermImportance()
        if config.doc_quality == "no":
            self.doc_quality = functional.AllOne()
        elif config.doc_quality == "epic":
            self.doc_quality = EPICDocQuality()

        self.norm = FunctionalFactory.get(config.norm)

    def forward(self, **kwargs):
        if "special_tokens_mask" in kwargs:
            kwargs.pop("special_tokens_mask")

        if "decoder_input_ids" not in kwargs:
            decoder_input_ids = kwargs["input_ids"]
            if isinstance(self.model, torch.nn.DataParallel):
                kwargs["decoder_input_ids"] = self.model.module._shift_right(decoder_input_ids)
            else:
                kwargs["decoder_input_ids"] = self.model._shift_right(decoder_input_ids)
            kwargs["decoder_input_ids"] = kwargs["decoder_input_ids"].to(decoder_input_ids.device)

        output = self.model(**kwargs, output_hidden_states=True)

        shifted_attention_mask = self.model._shift_right(kwargs["attention_mask"])
        logits = (
            output.logits
            * shifted_attention_mask.unsqueeze(-1)
        )
        # norm default: log(1+x)
        logits = self.norm(self.activation(logits))
        # (default: max) pooling over sequence tokens
        lex_weights = self.pool(logits)
        return SparseRep(dense=lex_weights)
    

    def build_model(self, model_name_or_dir):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_dir)
        return model