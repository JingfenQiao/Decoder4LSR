from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch
from torch import nn
from transformers import PretrainedConfig,AutoTokenizer, AutoModel,T5ForConditionalGeneration,AutoModelWithLMHead,AutoModelForSeq2SeqLM


class TransformerMLPEncoderDecoderSingleStepsConfig(PretrainedConfig):
    """
    Configuration for the TransformerEncoderDecoderSingleStepsConfig
    """

    model_type = "MLP_EncoderDecoder_SINGLESTEPS"

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


class TransformerMLPSparseEncoderDecoderSingleSteps(SparseEncoder):
    """
    TransformerSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLPEncoderDecoderSingleStepsConfig

    def __init__(self, config: TransformerMLPEncoderDecoderSingleStepsConfig = TransformerMLPEncoderDecoderSingleStepsConfig()):
        super().__init__(config)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.tf_base_model_name_or_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tf_base_model_name_or_dir
        )
        self.model = AutoModel.from_pretrained(config.tf_base_model_name_or_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.activation = FunctionalFactory.get(config.activation)
        self.norm = FunctionalFactory.get(config.norm)
        self.scale = nn.Parameter(torch.tensor(config.scale))


    def forward(self, to_scale=False, **kwargs):
        if "special_tokens_mask" in kwargs:
            kwargs.pop("special_tokens_mask")

        if "decoder_input_ids" not in kwargs:
            batch_size = kwargs["input_ids"].shape[0]
            # Initialize decoder input as the start token
            kwargs["decoder_input_ids"] = torch.tensor([[self.model.config.decoder_start_token_id]] * batch_size)
            kwargs["decoder_input_ids"] = kwargs["decoder_input_ids"].to(kwargs["input_ids"].device)

        output = self.model(**kwargs, output_hidden_states=True)
        decoder_last_hidden_state = output.decoder_hidden_states[-1]

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
