from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
model_name = "Qwen/Qwen2.5-7B-Instruct"   # 模型路径或model card

model = pipeline(
    task="text-generation",  # 指定文本生成任务  
    model=model_name, 
    model_kwargs={"torch_dtype": torch.bfloat16},  # 也可以设置auto
    device_map="auto"
)

prompt = "Give me a short introduction to large language model."

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
        messages,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=1,
    )

response = outputs[0]["generated_text"][-1]['content']


