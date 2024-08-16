import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(find_dotenv())

# 设置你的 OpenAI API 密钥
# openai.api_key = 'your-api-key-here'
openai = OpenAI()


def extract_company_names(text):
    # 定义提取公司名称的任务
    prompt = f"请提取以下文本中的公司名称：{text} 只输出文本中的公司名称，不要输出其他不相关的内容。如果文本中有多个公司名称，依次输出所有的公司名称，并用“、”分隔。"

    # 调用 GPT-4 模型
    response = openai.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
            {"role": "system", "content": "你是一个专业的数据标注人员。"},
            {"role": "user", "content": prompt}
        ]
    )

    # 获取 GPT-4 模型的回答
    company_names = response.choices[0].message.content
    # res = '、'.join(company_names.split('\n'))
    print("提取出的公司名称如下：")
    # print(res)
    print(company_names)
    return company_names


if __name__ == '__main__':
    # 定义需要处理的文本
    text = """
    Delta Apparel Inc Apparel Inc.，拥有Beachwear品牌Salt Life和运动品牌Soffe，近日申请破产，计划出售其资产。公司已同意将Salt Life品牌出售给Forager Capital Management LLC，交易金额约为2800万美元，作为一种“跟进出价”，即设置了底价，未来有可能接到更高报价。Delta Apparel Inc.于周日向特拉华州破产法院提交了第11章申请，资产约为3.378亿美元，负债总额为2.445亿美元。公司因棉花和其他原材料价格上涨以及产品需求减少而陷入财务困境。公司股价在盘前交易中下跌最多达37%，随后暂停交易。Delta Apparel Inc.在法院保护下将继续正常运营，并已与富国银行及其他现有贷款银行达成第11章融资协议，具体融资金额尚未披露。案件编号为24-11469，案件在特拉华州破产法院审理。
    """
    extract_company_names(text)

