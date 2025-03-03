import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config

openai.api_key = config.OPENAI_API_KEY

CHAT_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.9)
# prompt = ChatPromptTemplate.from_messages([
#     ("user", "{country}的首都在哪？")
# ])
# output_parser = StrOutputParser()
# chain = prompt | CHAT_LLM | output_parser
# res = chain.invoke({"country": "美国"})
# print(res)  # 美国的首都位于华盛顿特区（Washington, D.C.）。这是一个独立的联邦区，不属于任何一个州。
