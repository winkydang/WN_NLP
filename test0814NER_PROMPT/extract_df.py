import logging

import pandas as pd
import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(find_dotenv())

# 设置你的 OpenAI API 密钥
# openai.api_key = 'your-api-key-here'
openai = OpenAI()

# 配置日志记录
logging.basicConfig(filename='company_names.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


# 定义打印分割线的函数
def print_dashes(num=100):
    print()
    print('-' * num)
    print()


def extract_company_names(text):
    # 定义提取公司名称的任务
    prompt = f"请提取以下文本中的公司名称：{text} 只输出文本中的公司名称，不要输出其他不相关的内容。如果文本中有多个公司名称，依次输出所有的公司名称，并用“、”分隔。如果文本中没有公司名称，则输出""，不要输出其他内容。"

    # 调用 GPT-4 模型
    response = openai.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
            {"role": "system", "content": "你是一个专业的数据标注人员。"},
            {"role": "user", "content": prompt}
        ]
    )

    # 获取 gpt-4o-mini 模型的回答
    company_names = response.choices[0].message.content
    # res = '、'.join(company_names.split('\n'))

    # 打印原始文本
    print(f"原始文本：\n{text}")
    logging.info(f"原始文本：\n{text}")

    # 打印提取出的公司名称
    print("提取出的公司名称如下：")
    # print(res)
    print(company_names)
    print_dashes()

    # 将公司名称写入日志文件
    logging.info("提取出的公司名称如下：\n%s", company_names)
    logging.info(f"{'-' * 100}")
    return company_names


# 1. 读取CSV文件
# file_path = './data/0816_data_bloomberg.csv'
file_path = './data/data_bloomberg20240814155117.csv'
df = pd.read_csv(file_path)

# 5. 应用函数到CONTENT列
df['相关主体'] = df['content'].apply(extract_company_names)

# 6. 保存结果到新的CSV文件
# output_path = './data/output/result_0816_data_bloomberg.csv'
output_path = './data/output/result_data_bloomberg20240814155117.csv'
df.to_csv(output_path, index=False)

print(output_path)

