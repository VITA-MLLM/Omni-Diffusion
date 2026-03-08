# models.dream.__init__

from .modeling_dream import DreamModel
from .configuration_dream import DreamConfig  
from .tokenization_dream import DreamTokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

AutoConfig.register("Dream", DreamConfig)
AutoModelForCausalLM.register(DreamConfig, DreamModel)

DreamConfig.register_for_auto_class()
DreamModel.register_for_auto_class("AutoModelForCausalLM")
