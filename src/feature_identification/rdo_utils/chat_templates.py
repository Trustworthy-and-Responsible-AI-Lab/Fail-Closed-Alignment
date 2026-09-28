GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

QWEN2_CHAT_TEMPLATE = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

LLAMA2_CHAT_TEMPLATE = "[INST] {instruction} [/INST] "

GRANITE_CHAT_TEMPLATE = """<|start_of_role|>user<|end_of_role|>{instruction}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""

def apply_chat_template(model_path, instructions: list[str]):
    if 'llama-3' in model_path.lower() or 'llama3' in model_path.lower():
        prompts = [LLAMA3_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif 'qwen2' in model_path.lower():
        prompts = [QWEN2_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif 'llama-2' in model_path.lower() or 'llama2' in model_path.lower():
        prompts = [LLAMA2_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "gemma" in model_path.lower():
        prompts = [GEMMA_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "granite" in model_path.lower():
        prompts = [GRANITE_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    else:
        raise ValueError(f"Model {model_path} not supported, need to configure chat template")
    return prompts