import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, AutoConfig
from transformers import pipeline

# 1. 读取CSV文件
# file_path = '../data/data_bloomberg20240814155117.csv'
file_path = '../data/副本1副本.csv'
df = pd.read_csv(file_path)

# 2. 加载ERNIE模型和对应的分词器
model_name = "nghuyong/ernie-3.0-base-zh"
# model_name = "bert-base-chinese"
# model_name = "hfl/chinese-roberta-wwm-ext-large"

config = AutoConfig.from_pretrained(model_name)
print(config.id2label)  # 查看标签映射  # {0: 'LABEL_0', 1: 'LABEL_1'}

label_map = {0: "O", 1: "ORG"}  # 根据实际情况更新映射


tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)
model = BertForTokenClassification.from_pretrained(model_name)

# 3. 使用Hugging Face的pipeline进行命名实体识别
nlp = pipeline("ner", model=model, tokenizer=tokenizer)


# 4. 定义函数来提取公司实体
def extract_companies(text):
    ner_results = nlp(text)
    print(ner_results)  # 输出NER结果，查看标签和文本
    # companies = [result['word'] for result in ner_results if result['entity'] == 'B-ORG' or result['entity'] == 'I-ORG']
    companies = [result['word'] for result in ner_results if label_map[int(result['entity'].split('_')[-1])] == 'ORG']
    return '、'.join(companies)


# # 5. 应用函数到CONTENT列
# df['公司实体'] = df['content'].apply(extract_companies)
#
# # 6. 保存结果到新的CSV文件
# output_path = '../data/副本1副本_带公司实体_ERNIE.csv'
# df.to_csv(output_path, index=False)
#
# print(output_path)

test_text = "百度公司正在扩展其业务。"
results = extract_companies(test_text)
print()
print(f"results: {results}")


