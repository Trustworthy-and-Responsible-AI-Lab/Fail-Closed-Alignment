from feature_identification.dim_utils.pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:

    if 'qwen2' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.qwen2_model import Qwen2Model
        return Qwen2Model(model_path)
    if 'granite' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.granite_model import GraniteModel
        return GraniteModel(model_path)
    if 'llama-3' in model_path.lower() or 'llama3' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama-2' in model_path.lower() or 'llama2' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path) 
    elif 'mistral' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.mistral_model import MistralModel
        return MistralModel(model_path)
    elif 'falcon-3' in model_path.lower() or 'falcon3' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.falcon3_model import Falcon3Model
        return Falcon3Model(model_path)
    elif 'phi-3' in model_path.lower() or 'phi3' in model_path.lower():
        from feature_identification.dim_utils.pipeline.model_utils.phi3_model import Phi3Model
        return Phi3Model(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
