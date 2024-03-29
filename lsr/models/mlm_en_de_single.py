from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch
from torch import nn
from transformers import PretrainedConfig,AutoTokenizer, AutoModel,T5ForConditionalGeneration,AutoModelWithLMHead,AutoModelForSeq2SeqLM


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

class TransformerMLMEncoderDecoderSingleStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerEncoderDecoderSingleStepsConfig
    """

    model_type = "MLM_EncoderDecoder_SINGLESTEPS"

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
        Construct an instance of TransformerConfig
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


class TransformerMLMSparseEncoderDecoderSingleSteps(SparseEncoder):
    """
    TransformerSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLMEncoderDecoderSingleStepsConfig

    def __init__(self, config: TransformerMLMEncoderDecoderSingleStepsConfig = TransformerMLMEncoderDecoderSingleStepsConfig()):
        super(SparseEncoder, self).__init__(config)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.tf_base_model_name_or_dir
        )
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
            batch_size = kwargs["input_ids"].shape[0]
            # Initialize decoder input as the start token
            kwargs["decoder_input_ids"] = torch.tensor([[self.model.config.decoder_start_token_id]] * batch_size)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
            kwargs["decoder_input_ids"] = kwargs["decoder_input_ids"].to(device)


        output = self.model(**kwargs, output_hidden_states=True)
        decoder_last_hidden_state = output.decoder_hidden_states[-1]
        term_scores = self.term_importance(decoder_last_hidden_state)

        # remove padding tokens and special tokens
        logits = (
            output.logits
            * kwargs["attention_mask"].unsqueeze(-1)
            * term_scores
        )
        # norm default: log(1+x)
        logits = self.norm(self.activation(logits))
        # (default: max) pooling over sequence tokens
        lex_weights = self.pool(logits) 
        return SparseRep(dense=lex_weights)