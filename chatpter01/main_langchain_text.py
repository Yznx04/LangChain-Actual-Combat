from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(dotenv_path="../.env")

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.85,
    max_tokens=60,
)

messages = [
    {"role": "system", "content": "你是一个智能AI助手"},
    {"role": "user", "content": "请给我的花店起一个名字"},
]

response = llm.invoke(messages)
print(response)
