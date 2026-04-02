from transformers import AutoTokenizer, AutoModel, AutoProcessor,Qwen3VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration

def qwen25_vl_prompt(prompt):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True, padding_side="left")
    if '<image>' in prompt:
        prompt = prompt.replace('<image>', '')
    messages = [{"role": "user", "content": prompt.replace("{<|image_pad|>}", "<|vision_start|><|image_pad|><|vision_end|>")}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

def qwen3_nothink_prompt(prompt):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True, padding_side="left")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    # print(text)
    return text

def qwen3_think_prompt(prompt):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, padding_side="left")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return text

def qwen3_vl_nothink_prompt(prompt):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True, padding_side="left")
    if '<image>' in prompt:
        prompt = prompt.replace('<image>', '')
    messages = [{"role": "user", "content": prompt.replace("{<|image_pad|>}", "<|vision_start|><|image_pad|><|vision_end|>")}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    text += "<think>\n\n</think>\n\n"
    # print(text)
    return text

def intern_vl_prompt(prompt):
    tokenizer = AutoTokenizer.from_pretrained("internlm/InternVL2-8B", trust_remote_code=True, padding_side="left")
    if '<image>' in prompt:
        prompt = prompt.replace('<image>', '')
    messages = [{"role": "user", "content": prompt.replace("{<|image_pad|>}", "<image>\n")}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(text)
    return text

MODEL_CLASS = {
    "qwen2_5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl_nothink": Qwen3VLForConditionalGeneration,
    "qwen3vl_think": Qwen3VLForConditionalGeneration,
}

PROMPT_CLASS = {
    "qwen2_5vl": qwen25_vl_prompt,
    "qwen3_nothink": qwen3_nothink_prompt,
    "qwen3_think": qwen3_think_prompt,
    "qwen3vl_nothink": qwen3_vl_nothink_prompt,
    "intern_vl": intern_vl_prompt,
}

def get_model(model_name):
    if model_name in MODEL_CLASS:
        return MODEL_CLASS[model_name]
    else:
        return AutoModel

def build_prompt(prompt, model_name):
    return PROMPT_CLASS[model_name](prompt)
