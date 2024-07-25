import json

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")

train_data = dataset['train']

# 方式1：先转成 dataframe 类型，再将df存成json文件
df_train = pd.DataFrame(train_data)
df_train.to_json('./data/train_sem_1.json', orient='records', lines=True)

df_train.to_csv('./data/train_sem_1.csv', index=False)
print()



