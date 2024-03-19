import os.path
import re

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from zhon.hanzi import punctuation

from config import BASE_DIR


# def cleaned_text(text):  # 清洗数据
#     # 去除非中文字符（保留中文、数字和中文标点）
#     cleaned_text = re.sub(f"[^\u4e00-\u9fa5{punctuation}0-9]", "", text)
#     # # 去除HTML标签
#     # cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
#     # # 去除特殊字符
#     # cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
#     # 分句。这里简单使用 中文的 句号、问号和叹号 作为分句符号
#     # sentences = re.split(f"([。？！])", cleaned_text)
#     sentences = re.split(r'(?<=[。？！])', cleaned_text)
#     return sentences
#
#
# df = pd.read_csv(os.path.join(BASE_DIR, 'yq_cls/tmp/data/yq_50w.csv'))
#
# col = ['EMORATE', 'CONTENT']  # 存储需要用到的两列
#
# df_save = df[col]  # 创建新的DATAFRAME，只包含需要用到的两列
#
# # 应用清洗函数到 context 列
# df_save.loc[:, 'CONTENT_CLEANED'] = df_save['CONTENT'].apply(cleaned_text)
# # df_save['CONTENT_CLEANED'] = df_save['CONTENT'].apply(cleaned_text)
# # print(df_save.head())
#
# # # 分割数据集，这里的test_size=0.2代表测试集占20%，训练集占80%
# # train_data, test_data = train_test_split(df_save[['EMORATE', 'CONTENT_CLEANED']], shuffle=True, test_size=0.2, random_state=42)
#
# # 定义训练集和测试集的保存路径
# # train_file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/train.csv')
# # test_file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/test.csv')
# yq_file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/yq_clear_50w.csv')
#
# # train_data.to_csv(train_file_path, index=False, header=False)  # index=False, header=False，不要索引，不要列名
# # test_data.to_csv(test_file_path, index=False, header=False)
# df_save.to_csv(yq_file_path, index=False)  # 保存全部数据，包括进行分句的和未进行分句的列，保留列名



# 数据预处理之后，分割训练集和验证集
df = pd.read_csv(os.path.join(BASE_DIR, 'yq_cls/tmp/data/yq_clear_50w.csv'))
# 分割数据集，这里的test_size=0.2代表测试集占20%，训练集占80%
train_data, test_data = train_test_split(df, shuffle=True, test_size=0.2, random_state=42)

# 定义训练集和测试集的保存路径
train_file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/train.csv')
test_file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/test.csv')

train_data.to_csv(train_file_path, index=False, header=False)  # index=False, header=False，不要索引，不要列名
test_data.to_csv(test_file_path, index=False, header=False)
