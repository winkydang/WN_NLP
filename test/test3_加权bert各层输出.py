import torch
from transformers import BertModel, BertTokenizer

# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True)

# 输入文本
text = "This is a sample text for BERT"
inputs = tokenizer(text, return_tensors='pt')

# 前向传播，获取所有层的隐藏状态
outputs = model(**inputs)
hidden_states = outputs.hidden_states  # 这是一个包含13个元素的元组，第一个元素是embedding层的输出，后面12个元素是transformer层的输出

# 初始化权重
weights = torch.tensor([1/12] * 12)  # 这里我们简单地初始化为均匀分布的权重

# 将所有transformer层的输出按权重相加
weighted_sum = torch.zeros_like(hidden_states[0])
for i, hidden_state in enumerate(hidden_states[1:]):  # 跳过第一个元素（embedding层的输出）
    weighted_sum += weights[i] * hidden_state

# 获取最终的结果向量
result_vector = weighted_sum.mean(dim=0)  # 可以根据需求选择不同的聚合方式，这里用的是mean

print(result_vector)
