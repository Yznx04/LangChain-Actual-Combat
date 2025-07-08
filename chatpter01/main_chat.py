from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com/")

response = client.chat.completions.create(
    model="deepseek-chat",
    temperature=0.5,
    max_tokens=100,
    messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起一个名"}
    ]
)
print(response.choices[0].json())
