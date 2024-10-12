# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

llm_chain = model.eval()

llm_chain.generation_config = GenerationConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

text = "你好"

messages = [
    {"role": "user", "content": text}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize="True", add_generation_prompt=True, return_tensor='pt')

output_ids = llm_chain.generate(input_ids, max_new_tokens=4098)

response = tokenizer.batch_decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)



