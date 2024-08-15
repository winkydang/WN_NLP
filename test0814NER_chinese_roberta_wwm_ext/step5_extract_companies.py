import json

import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from test0814NER_chinese_roberta_wwm_ext.step4_模型测试 import post_processing

# 1. 读取CSV文件
# file_path = './data/资讯data/data_bloomberg20240814155117.csv'
file_path = './data/资讯data/副本1副本.csv'
df = pd.read_csv(file_path)


labels_path = "./data/processed/labels.json"
model_name = './output'
# max_length = 300
max_length = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载label
labels_map = {}
with open(labels_path, "r", encoding="utf-8") as r:
    labels = json.loads(r.read())
    for label in labels:
        label_id = labels[label]
        labels_map[label_id] = label

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels_map))
model.to(device)


# 4. 定义函数来提取公司实体
def extract_companies(text):
    encoded_input = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    input_ids = torch.tensor([encoded_input['input_ids']]).to(device)
    attention_mask = torch.tensor([encoded_input['attention_mask']]).to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    result = post_processing(outputs, text, labels_map)
    print(f"text: {text}")
    print(f"result: {result}")
    company_result = result['company'] if 'company' in result.keys() else []
    # government_result = result['government'] if 'government' in result.keys() else []
    # organization_result = result['organization'] if 'organization' in result.keys() else []

    # res = company_result + ',' + government_result + ',' + organization_result
    res = []
    res.extend(company_result)  # res.extend()返回的时是None，res原地修改
    # res.extend(government_result)
    # res.extend(organization_result)
    print(f"res: {res}")
    # print(f"company_result: {company_result}")
    dashes = '-' * 50
    print(dashes)
    print()
    # return company_result
    # return '、'.join(list(company_result))
    print(f"res_str: {'、'.join(res)}")
    return '、'.join(res)


# 5. 应用函数到CONTENT列
df['公司实体'] = df['content'].apply(extract_companies)

# 6. 保存结果到新的CSV文件
output_path = './data/资讯data/res/副本1副本_带公司实体_ERNIE.csv'
df.to_csv(output_path, index=False)

print(output_path)

# test_text = "百度公司正在扩展其业务。"
# results = extract_companies(test_text)
# print()
# print(f"results: {results}")


