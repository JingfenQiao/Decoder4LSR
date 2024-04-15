from .cls_mlm import TransformerCLSMLPSparseEncoder, TransformerCLSMLMConfig
from .mlm import (
    TransformerMLMSparseEncoder,
    TransformerMLMConfig,
)
from .mlp import TransformerMLPSparseEncoder, TransformerMLPConfig
from .binary import BinaryEncoder, BinaryEncoderConfig
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, AutoModel

from lsr.models.sparse_encoder import SparseEncoder


class DualSparseConfig(PretrainedConfig):
    model_type = "DualEncoder"

    def __init__(self, shared=False, freeze_doc_encoder=False, **kwargs):
        self.shared = shared
        self.freeze_doc_encoder = freeze_doc_encoder
        super().__init__(**kwargs)


class DualSparseEncoder(PreTrainedModel):
    """
    DualSparseEncoder class that encapsulates encoder(s) for query and document.

    Attributes
    ----------
    shared: bool
        to use a shared encoder for both query/document
    encoder: lsr.models.SparseEncoder
        a shared encoder for encoding both queries and documnets. This encoder is used only if 'shared' is True, otherwise 'None'
    query_encoder: lsr.models.SparseEncoder
        a separate encoder for encoding queries, only if 'shared' is False
    doc_encoder: lsr.models.SparseEncoder
        a separate encoder for encoding documents, only if 'shared' is False
    Methods
    -------
    from_pretrained(model_dir_or_name: str)
    """

    config_class = DualSparseConfig

    def __init__(
        self,
        query_encoder: SparseEncoder,
        doc_encoder: SparseEncoder = None,
        config: DualSparseConfig = DualSparseConfig(),
    ):
        super().__init__(config)
        self.config = config
        if self.config.shared:
            self.encoder = query_encoder
        else:
            if isinstance(query_encoder, str):
                self.query_encoder = AutoModel.from_pretrained(query_encoder)
            else:
                self.query_encoder = query_encoder
            if isinstance(doc_encoder, str):
                self.doc_encoder = AutoModel.from_pretrained(doc_encoder)
            else:
                self.doc_encoder = doc_encoder
            if self.config.freeze_doc_encoder:
                for param in self.doc_encoder.parameters():
                    param.requires_grad = False

    def encode_queries(self, to_dense=True, **queries):
        """
        Encode a batch of queries with the query encoder
        Arguments
        ---------
        to_dense: bool
            If True, return the output vectors in dense format; otherwise, return in the sparse format (indices, values)
        queries:
            Input dict with {"input_ids": torch.Tensor, "attention_mask": torch.Tensor , "special_tokens_mask": torch.Tensor }
        """
        if to_dense:
            if self.config.shared:
                return self.encoder(**queries).to_dense(reduce="sum")
            else:
                return self.query_encoder(**queries).to_dense(reduce="sum")
        else:
            if self.config.shared:
                return self.encoder(**queries)
            else:
                return self.query_encoder(**queries)

    def encode_docs(self, to_dense=True, **docs):
        """
        Encode a batch of documents with the document encoder
        """
        if to_dense:
            if self.config.shared:
                return self.encoder(**docs).to_dense(reduce="amax")
            else:
                return self.doc_encoder(**docs).to_dense(reduce="amax")
        else:
            if self.config.shared:
                return self.encoder(**docs)
            else:
                return self.doc_encoder(**docs)

    def forward(self, loss, queries, doc_groups, labels=None, **kwargs):
        """Compute the loss given (queries, docs, labels)"""
        q_reps = self.encode_queries(**queries)
        doc_group_reps = [self.encode_docs(
            **doc_group, to_dense=True) for doc_group in doc_groups]
        # self.encode_docs(**doc_groups)
        if labels is None:
            output = loss(q_reps, *doc_group_reps)
        else:
            output = loss(q_reps, *doc_group_reps, labels)
        return output

    # def save_pretrained(self, model_dir):
    #     """Save both query and document encoder"""
    #     self.config.save_pretrained(model_dir)
    #     if self.config.shared:
    #         self.encoder.save_pretrained(model_dir + "/shared_encoder")
    #     else:
    #         self.query_encoder.save_pretrained(model_dir + "/query_encoder")
    #         self.doc_encoder.save_pretrained(model_dir + "/doc_encoder")

    # @classmethod
    # def from_pretrained(cls, model_dir_or_name):
    #     """Load query and doc encoder from a directory"""
    #     config = DualSparseConfig.from_pretrained(model_dir_or_name)
    #     if config.shared:
    #         shared_encoder = AutoModel.from_pretrained(
    #             model_dir_or_name + "/shared_encoder"
    #         )
    #         return cls(shared_encoder, config=config)
    #     else:
    #         query_encoder = AutoModel.from_pretrained(
    #             model_dir_or_name + "/query_encoder"
    #         )
    #         doc_encoder = AutoModel.from_pretrained(
    #             model_dir_or_name + "/doc_encoder")
    #         return cls(query_encoder, doc_encoder, config)

    def save_pretrained(self, model_dir):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir)
        if self.config.shared:
            self.encoder.save_pretrained(model_dir + "/shared_encoder")
            if self.config.lora:
                self.encoder.model.save_pretrained(model_dir + "/shared_encoder",save_embedding_layers=True)
        else:
            self.query_encoder.save_pretrained(model_dir + "/query_encoder")
            self.doc_encoder.save_pretrained(model_dir + "/doc_encoder")

    @classmethod
    def from_pretrained(cls, model_dir_or_name):
        """Load query and doc encoder from a directory"""
        config = DualSparseConfig.from_pretrained(model_dir_or_name)
        # config.lora = False
        print(config)
        if config.shared:
            if config.lora:
                model_config = AutoConfig.from_pretrained(model_dir_or_name + "/shared_encoder")
                model_config.lora_pretrianed = True
                shared_encoder = AutoModel.from_config(model_config)
            else:
                shared_encoder = AutoModel.from_pretrained(
                    model_dir_or_name + "/shared_encoder"
                )

            return cls(shared_encoder, config=config)
        else:
            query_encoder = AutoModel.from_pretrained(
                model_dir_or_name + "/query_encoder"
            )
            doc_encoder = AutoModel.from_pretrained(model_dir_or_name + "/doc_encoder")
            return cls(query_encoder, doc_encoder, config)

from .mlm_en_multi import TransformerMLMEncoderOnlyConfig, TransformerMLMEncoderOnlySparseEncoder
from .mlm_de_multi_t5 import TransformerMLMT5DecoderMultiStepsConfig, TransformerMLMSparseT5DecoderMultiSteps
from .mlm_de_multi_opt import TransformerMLMOPTDecoderMultiStepsConfig, TransformerMLMSparseOPTDecoderMultiSteps
from .mlm_en_de_single import TransformerMLMEncoderDecoderSingleStepsConfig, TransformerMLMSparseEncoderDecoderSingleSteps
from .mlm_en_de_multi import TransformerMLMEncoderDecoderMultiStepsConfig, TransformerMLMSparseEncoderDecoderMultiSteps
from .mlm_de_multi_opt_lora import TransformerMLMSparseOPTLoraDecoderMultiSteps, TransformerMLMOPTLoraDecoderMultiStepsConfig
from .mlm_en_de_multi_multi_decoding import TransformerMLMSparseEncoderDecoderMultiStepsMultiSteps, TransformerMLMEncoderDecoderMultiStepsMultiStepsConfig

from .mlp_en_multi import TransformerMLPEncoderOnlyConfig, TransformerMLPEncoderOnlySparseEncoder
from .mlp_de_multi_t5 import TransformerMLPT5DecoderMultiStepsConfig, TransformerMLPSparseT5DecoderMultiSteps
from .mlp_de_multi_opt import TransformerMLPOPTDecoderMultiStepsConfig, TransformerMLPSparseOPTDecoderMultiSteps
from .mlp_en_de_single import TransformerMLPEncoderDecoderSingleStepsConfig, TransformerMLPSparseEncoderDecoderSingleSteps
from .mlp_en_de_multi import TransformerMLPEncoderDecoderMultiStepsConfig, TransformerMLPSparseEncoderDecoderMultiSteps
from .mlp_de_multi_opt_lora import TransformerMLPSparseOPTLoraDecoderMultiSteps, TransformerMLPOPTLoraDecoderMultiStepsConfig


AutoConfig.register("BINARY", BinaryEncoderConfig)
AutoModel.register(BinaryEncoderConfig, BinaryEncoder)
AutoConfig.register("MLP", TransformerMLPConfig)
AutoModel.register(TransformerMLPConfig, TransformerMLPSparseEncoder)
AutoConfig.register("MLM", TransformerMLMConfig)
AutoModel.register(TransformerMLMConfig, TransformerMLMSparseEncoder)
AutoConfig.register("CLS_MLM", TransformerCLSMLMConfig)
AutoModel.register(TransformerCLSMLMConfig, TransformerCLSMLPSparseEncoder)

AutoConfig.register("MLM_ENCODER_ONLY", TransformerMLMEncoderOnlyConfig)
AutoModel.register(TransformerMLMEncoderOnlyConfig, TransformerMLMEncoderOnlySparseEncoder)

AutoConfig.register("MLM_T5_DECODER_MULTISTEPS", TransformerMLMT5DecoderMultiStepsConfig)
AutoModel.register(TransformerMLMT5DecoderMultiStepsConfig, TransformerMLMSparseT5DecoderMultiSteps)

AutoConfig.register("MLM_OPT_DECODER_MULTISTEPS", TransformerMLMOPTDecoderMultiStepsConfig)
AutoModel.register(TransformerMLMOPTDecoderMultiStepsConfig, TransformerMLMSparseOPTDecoderMultiSteps)

AutoConfig.register("MLM_OPT_LORA_DECODER_MULTISTEPS", TransformerMLMOPTLoraDecoderMultiStepsConfig)
AutoModel.register(TransformerMLMOPTLoraDecoderMultiStepsConfig, TransformerMLMSparseOPTLoraDecoderMultiSteps)

AutoConfig.register("MLM_EncoderDecoder_SINGLESTEPS", TransformerMLMEncoderDecoderSingleStepsConfig)
AutoModel.register(TransformerMLMEncoderDecoderSingleStepsConfig, TransformerMLMSparseEncoderDecoderSingleSteps)

AutoConfig.register("MLM_EncoderDecoder_MULTISTEPS", TransformerMLMEncoderDecoderMultiStepsConfig)
AutoModel.register(TransformerMLMEncoderDecoderMultiStepsConfig, TransformerMLMSparseEncoderDecoderMultiSteps)

AutoConfig.register("MLM_EncoderDecoder_MULTISTEPS_MULTISTEPS", TransformerMLMEncoderDecoderMultiStepsMultiStepsConfig)
AutoModel.register(TransformerMLMEncoderDecoderMultiStepsMultiStepsConfig, TransformerMLPSparseOPTLoraDecoderMultiSteps)

AutoConfig.register("MLP_ENCODER_ONLY", TransformerMLPEncoderOnlyConfig)
AutoModel.register(TransformerMLPEncoderOnlyConfig, TransformerMLPEncoderOnlySparseEncoder)

AutoConfig.register("MLP_T5_DECODER_MULTISTEPS", TransformerMLPT5DecoderMultiStepsConfig)
AutoModel.register(TransformerMLPT5DecoderMultiStepsConfig, TransformerMLPSparseT5DecoderMultiSteps)

AutoConfig.register("MLP_OPT_DECODER_MULTISTEPS", TransformerMLPOPTDecoderMultiStepsConfig)
AutoModel.register(TransformerMLPOPTDecoderMultiStepsConfig, TransformerMLPSparseOPTDecoderMultiSteps)

AutoConfig.register("MLP_OPT_LORA_DECODER_MULTISTEPS", TransformerMLPOPTLoraDecoderMultiStepsConfig)
AutoModel.register(TransformerMLPOPTLoraDecoderMultiStepsConfig, TransformerMLPSparseOPTLoraDecoderMultiSteps)

AutoConfig.register("MLP_EncoderDecoder_SINGLESTEPS", TransformerMLPEncoderDecoderSingleStepsConfig)
AutoModel.register(TransformerMLPEncoderDecoderSingleStepsConfig, TransformerMLPSparseEncoderDecoderSingleSteps)

AutoConfig.register("MLP_EncoderDecoder_MULTISTEPS", TransformerMLPEncoderDecoderMultiStepsConfig)
AutoModel.register(TransformerMLPEncoderDecoderMultiStepsConfig, TransformerMLPSparseEncoderDecoderMultiSteps)