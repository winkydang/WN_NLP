from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

from configs.llm import CHAT_LLM

# model = HfApiModel()
model = CHAT_LLM

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")