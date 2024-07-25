"""
reference link: https://colab.research.google.com/drive/1pAOYaaVZWY5abusxw9ar3uXiTobK-8mi?usp=sharing#scrollTo=kdx9wI0ZdVZr

Training Mistral-7b AI on a Single GPU using PEFT LORA with Google Colab.
Welcome to this notebook that will show you how to finetune Mistral-7b using the recent peft library and bitsandbytes for loading large models in 4-bit.

The fine-tuning method will rely on a recent method called "Low Rank Adapters" (LoRA), instead of fine-tuning the entire model you just have to fine-tune these adapters and load them properly inside the model. After fine-tuning the model you can also share your adapters on the 🤗 Hub and load them very easily. Let's get started!

Note that this could be used for any model that supports device_map (i.e. loading the model with accelerate).

# This is a demo code to showcase how to use the PEFT integration with HF transformers.
"""


def get_completion(query: str, model, tokenizer) -> str:
    """
    define a wrapper function which will get completion from the model from a user question
    :param query:
    :param model:
    :param tokenizer:
    :return:
    """
    device = "cuda:0"
    prompt_template = """
        Below is an instruction that describes a task. Write a response that appropriately complttes the request.
        ### Question:
        {query}
        
        ### Answer:
    """
    prompt = prompt_template.format(query=query)

    encodes = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodes.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return (decoded[0])


# step1:install necessary packages
# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q datasets


# step2: model loading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# specify the model ID and load it with our previously define quantization configuration
model_id = "mistralai/Mistral-7B-v0.1"

# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

# run a inference on the base model.  The model does not seem to understand our instruction and gives us a list of questions related to our query.
# result = get_completion(query="Will capital gains affect my tax bracket?", model=model, tokenizer=tokenizer)
# print(result)

# step3: Load dataset for finetuning
# let's load a dataset on fiannce, to finetune our model on basic finance knowledges.
from datasets import load_dataset

data = load_dataset("gbharti/finance-alpaca", split="train")

# Explore the data_1
df = data.to_pandas()
print(df.head(10))

# Instruction Fintuning - Prepare the dataset under the format of "prompt" so the model can better understand :
#
# the function generate_prompt : take the instruction and output and generate a prompt
# shuffle the dataset
# tokenizer the dataset

def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    # Samples with additional context into.
    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'

    # Without
    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'
    return text

# add the "prompt" column in the dataset
text_column = [generate_prompt(data_point) for data_point in data]
data = data.add_column("prompt", text_column)

# tokenize our data_1 so the model can understand.
data = data.shuffle(seed=1234)  # Shuffle dataset here
data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

# Split dataset into 90% for training and 10% for testing
data = data.train_test_split(test_size=0.1)
train_data = data["train"]
test_data = data["test"]

print(test_data)

# Step 4 - Apply Lora
# Here comes the magic with peft! Let's load a PeftModel and specify that we are going to use low-rank adapters (LoRA) using get_peft_model utility function and the prepare_model_for_kbit_training method from PEFT.
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
print_trainable_parameters(peft_model)


# Add adapter to the Model

model.add_adapter(lora_config, adapter_name="adapter")

# Step 5 - Run the training!
from huggingface_hub import notebook_login
notebook_login()

# Setting the training arguments:
#
# for the reason of demo, we just ran it for few steps (100) just to showcase how to use this integration with existing tools on the HF ecosystem.
# from datasets import load_dataset
# data_1 = load_dataset("ronal999/finance-alpaca-demo", split='train')
# data_1 = data_1.train_test_split(test_size=0.1)
# train_data = data_1["train"]
# test_data = data_1["test"]

# import transformers

# tokenizer.pad_token = tokenizer.eos_token


# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=train_data,
#     eval_dataset=test_data,
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         warmup_steps=0.03,
#         max_steps=100,
#         learning_rate=2e-4,
#         fp16=True,
#         logging_steps=1,
#         output_dir="outputs_mistral_b_finance_finetuned_test",
#         optim="paged_adamw_8bit",
#         save_strategy="epoch",
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )


# !pip install -q trl
# Here I reload the model and specify it should be loaded in a single GPU to avoid errors" Expected all tensors to be on the same device,
# but found at least two devices, cuda:0 and cpu! when resuming training"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

#new code using SFTTrainer
import transformers
from trl import SFTTrainer

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start the training
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# Share adapters on the 🤗 Hub
model.push_to_hub("mistral_b_finance_finetuned_test")
tokenizer.push_to_hub("mistral_b_finance_finetuned_test")

# Step 6 Evaluating the model qualitatively: run an inference!


# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q datasets

# Load directly adapters from the Hub using the command below
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "Ronal999/mistral_b_finance_finetuned_test"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# You can then directly use the trained model that you have loaded from the 🤗 Hub for inference as you would do it usually in transformers.
result = get_completion(query="Will capital gains affect my tax bracket?", model=model, tokenizer=tokenizer)
print(result)




