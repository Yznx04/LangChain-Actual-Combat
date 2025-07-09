from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com/beta")

response = client.completions.create(
    model="deepseek-chat",
    temperature=0.5,
    max_tokens=100,
    prompt="请给我的花店起个名字"
)
print(response.choices[0].text.strip())
