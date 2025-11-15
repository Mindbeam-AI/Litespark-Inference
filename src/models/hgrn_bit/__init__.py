# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hgrn_bit import HGRNBitConfig
from .modeling_hgrn_bit import HGRNBitForCausalLM, HGRNBitModel

AutoConfig.register(HGRNBitConfig.model_type, HGRNBitConfig)
AutoModel.register(HGRNBitConfig, HGRNBitModel)
AutoModelForCausalLM.register(HGRNBitConfig, HGRNBitForCausalLM)


__all__ = ['HGRNBitConfig', 'HGRNBitForCausalLM', 'HGRNBitModel']
