from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
# from datasets import load_dataset
#
# dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
#
# labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
# id2label = {idx: label for idx, label in enumerate(labels)}
# label2id = {label: idx for idx, label in enumerate(labels)}
# print(id2label)
id2label = {0: 'anger', 1: 'anticipation', 2: 'disgust', 3: 'fear', 4: 'joy', 5: 'love', 6: 'optimism', 7: 'pessimism', 8: 'sadness', 9: 'surprise', 10: 'trust'}

model_path = './bert-finetuned-sem_eval-english/checkpoint-4275'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 初始化一个分词器，来处理输入文本
tokenizer = BertTokenizer.from_pretrained(model_path)  # 初始化一个分词器，来处理输入文本
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

text = "I'm happy, the bert model is so good"  # ['joy', 'love', 'optimism']
# text = "I'm hungry"  # ['sadness']
# text = "I'm angry"  # ['anger', 'disgust']
encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(model.device) for k, v in encoding.items()}

outputs = model(**encoding)
logits = outputs.logits
print(logits.shape)
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)




