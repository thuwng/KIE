from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

from .configuration_layoutlmv3 import LayoutLMv3Config
from .modeling_layoutlmv3 import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Model,
)
from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast


def _safe_register(call, mapping, key, value):
    """Đăng ký lớp của layoutlmft vào Auto*, chạy được trên mọi bản transformers.
    1) bản mới:  register(..., exist_ok=True)
    2) bản cũ chưa có exist_ok (TypeError): register(...) thường
    3) transformers đã có sẵn 'layoutlmv3' và không cho ghi đè (ValueError): ghi thẳng vào
       mapping._extra_content -- transformers tra _extra_content TRƯỚC bản có sẵn,
       nên Auto* luôn trả về lớp của layoutlmft (bản có max_layers / spatial / attention bias)."""
    try:
        call(exist_ok=True)
        return
    except TypeError:
        pass
    try:
        call()
    except ValueError:
        mapping._extra_content[key] = value


_safe_register(lambda **kw: AutoConfig.register("layoutlmv3", LayoutLMv3Config, **kw),
               CONFIG_MAPPING, "layoutlmv3", LayoutLMv3Config)
for _auto, _model in ((AutoModel, LayoutLMv3Model),
                      (AutoModelForTokenClassification, LayoutLMv3ForTokenClassification),
                      (AutoModelForQuestionAnswering, LayoutLMv3ForQuestionAnswering),
                      (AutoModelForSequenceClassification, LayoutLMv3ForSequenceClassification)):
    _safe_register(lambda _a=_auto, _m=_model, **kw: _a.register(LayoutLMv3Config, _m, **kw),
                   _auto._model_mapping, LayoutLMv3Config, _model)
_safe_register(lambda **kw: AutoTokenizer.register(LayoutLMv3Config, slow_tokenizer_class=LayoutLMv3Tokenizer,
                                                   fast_tokenizer_class=LayoutLMv3TokenizerFast, **kw),
               TOKENIZER_MAPPING, LayoutLMv3Config, (LayoutLMv3Tokenizer, LayoutLMv3TokenizerFast))
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})