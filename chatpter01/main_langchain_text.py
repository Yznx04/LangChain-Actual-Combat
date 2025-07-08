from dotenv import load_dotenv
from langchain.llms import OpenAIChat

load_dotenv(dotenv_path="../.env")

llm = OpenAIChat(
    openai_api_base="https://api.deepseek.com/beta",
    model="deepseek-chat",
    temperature=0.85,
    max_tokens=60,
)
response = llm.invoke("请给我的花店起一个名字")
print(response)
