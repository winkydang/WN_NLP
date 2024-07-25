from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")


# print(dataset['train'][0])
# print(dataset['test'][0])
# print(dataset['validation'][0])
# print(dataset)

labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def preprocess_data(examples):
    text = examples["Tweet"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
print()

for example in encoded_dataset['train']:
    [id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]
encoded_dataset.set_format("torch")  # 将数据集的格式设置为 PyTorch tensors，以便与 PyTorch 模型兼容。
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",  problem_type="multi_label_classification",  num_labels=len(labels), id2label=id2label, label2id=label2id)
batch_size = 8  # 修改此参数进行训练
metric_name = "f1"
args = TrainingArguments(f"bert-finetuned-sem_eval-english", evaluation_strategy="epoch", save_strategy = "epoch", learning_rate=2e-5, per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,num_train_epochs=5, weight_decay=0.01,load_best_model_at_end=True,metric_for_best_model=metric_name,)
# f"bert-finetuned-sem_eval-english"：存放模型输出和日志的文件夹名称

def multi_label_metrics(predictions, labels, threshold=0.5):  # 多标签评估指标
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


trainer = Trainer(model,args,train_dataset=encoded_dataset["train"],eval_dataset=encoded_dataset["validation"],tokenizer=tokenizer,compute_metrics=compute_metrics)
trainer.train()


text = "I'm happy, the bert model is so good"
encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
outputs = trainer.model(**encoding)
logits = outputs.logits
print(logits.shape)
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)


