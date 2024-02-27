import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.nn import BCEWithLogitsLoss


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


# 初始化Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 假设max_len为128
max_len = 128
# 假设你有texts和labels
texts = ["example sentence 1", "example sentence 2"]
labels = [[1, 0, 1], [0, 1, 0]]  # 假设是3个可能的标签

# 创建数据集和数据加载器
dataset = TextDataset(texts, labels, tokenizer, max_len)
loader = DataLoader(dataset, batch_size=2)

# 模型定义
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# 训练准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

for batch in loader:
    optimizer.zero_grad()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)

    loss_fn = BCEWithLogitsLoss()
    loss = loss_fn(outputs.logits, labels)

    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")
