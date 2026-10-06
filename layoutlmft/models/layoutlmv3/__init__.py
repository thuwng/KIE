from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

from .configuration_layoutlmv3 import LayoutLMv3Config
from .modeling_layoutlmv3 import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Model,
)
from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast


AutoConfig.register("layoutlmv3", LayoutLMv3Config, exist_ok=True)
AutoModel.register(LayoutLMv3Config, LayoutLMv3Model, exist_ok=True)
AutoModelForTokenClassification.register(LayoutLMv3Config, LayoutLMv3ForTokenClassification, exist_ok=True)
AutoModelForQuestionAnswering.register(LayoutLMv3Config, LayoutLMv3ForQuestionAnswering, exist_ok=True)
AutoModelForSequenceClassification.register(LayoutLMv3Config, LayoutLMv3ForSequenceClassification, exist_ok=True)
AutoTokenizer.register(
    LayoutLMv3Config, 
    slow_tokenizer_class=LayoutLMv3Tokenizer, 
    fast_tokenizer_class=LayoutLMv3TokenizerFast,
    exist_ok=True
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})