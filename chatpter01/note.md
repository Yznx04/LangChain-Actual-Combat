# LangChain安装与快速入门

## 快速安装LangChain

使用uv进行安装：

```text
uv add langchain
```

使用pip进行安装

```
pip install langchain
```

## 使用OpenAI调用模型

LangChain本质上就是对各种大模型的API的套壳，是为了方便我们使用这些API，搭建起来的一些框架、模块和接口。

### 安装OpenAI

```
uv add openai
```

### 调用text模型

首先安装`openai`， 使用uv安装

```
uv add openai
```

然后需要导入`OpenAI的key`， 这里使用deepseek的key, DeepSeek是兼容OpenAI的格式的, 最好是通过环境变量来设置key：

``` python
import os

os.environ["OPENAI_API_KEY"] = "xxx"
```

或者是通过`.env`文件，通过`python-dotenv`库来加载

```python
from dotenv import load_dotenv

load_dotenv()
```

然后是使用：

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com/beta")

response = client.completions.create(
    model="deepseek-chat",
    temperature=0.5,
    max_tokens=100,
    prompt="请给我的花店起个名字"
)
print(response.choices[0].text.strip())
```

这里需要注意的是，因为使用的是deepseek的地址，所以需要指定base_url的值为deepseek的值，这个可以参考deepseek的文档。

OpenAI常见参数：

1. model：模型的类型
2. prompt：提示，也就是输入给模型的问题或者提示，告诉模型它们要它做什么
3. temperature：参数影响模型输出的随机性。值越高（接近1），输出就越随机;值越低（接近0），输出就越稳定
4. max_tokens: 限制模型输出的最大长度，这个长度是以Tokens为单位
5. suffix：参数允许用户为模型生成的输出文本后附加一个后缀。默认值为null
6. top_p: 使用核心抽样。模型将只考虑概率质量最高的Tokens
7. n: 这个参数决定了为每个提示生成多少个完整的输出。
8. stream：参数决定了是否实时流式传输生产的Tokens.如果设置为True,则Token将在生成时被发送
9. logprobs：参数要求包括最有可能的Tokens的对数概率。
10. echo：如果设置为True,除了生成完整的内容外，还会回显提示
11. stop：允许你指定一个或多个序列，当模型遇到这些序列时，它会停止生成Tokens,返回的文本将不包含stop序列
12. presence_penalty: 一个在-2.0和2.0之间的数字，正值会惩罚已经出现在文本中新的Tokens,这可以帮助模型更多地谈论新话题
13. frequency_penalty: 一个在-2.0和2.0之间的数字，正值会惩罚到目前为止在文本中频繁出现的Tokens,这可以减少模型重复相同内容的可能性
14. best_of: 参数会在服务器生成best_of个完整输出，并返回其中最好的一个（即每个Token的对数概率最高的那个）
15. logit_bias: 修改制定Tokens在完成中出现的可能性，接受一个Json对象，该对象对Tokens映射到-100到100之间的偏置值
16. user:可选参数，表示你的最终用户的唯一标识符，可以帮助OpenAI检测和检测滥用

OpenAI的返回一个响应对象，该对象包含了模型生成的输出和一些信息，主要字段包括：

1. id：响应的唯一标识符
2. object：表示该响应的对象类型
3. created：表示响应生成的时间
4. model：表示生成响应的模型的名称
5. choices：一个列表，其中包含了模型生成的所有输出，除非指定了n参数，否则通常只包含一个条目，choices中的内容是text
6. usage：提供了关于文本生成过程的统计信息，包括prompt_tokens(提示的Token数量), completion_tokens(生成的Token数量), total_tokens(总的Token数量)

choices字段是一个列表，其中每一个选择都是一个字典，主要包括：

- text：模型生成的文本
- finish_reason: 模型停止生成的原因，可能的值包括stop（遇到了停止标记）、length（达到了最大长度）或temperature（根据设定的温度参数决定停止）

### 调用chat模型

Chat模型与Text模型调用类型，但是需要加上chat

```python
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
```

有两个专属于Chat模型的概念：

- 消息： 传入模型的提示。此处的messages是一个参数列表，包含了多个消息，每个消息由
   - role：角色，可选参数：system、user、assistant
     - system：系统消息主要用于设定对话的背景或上下文。这可以帮助模型理解它在对话中的角色和任务。
     - user：用户消息是从用户或人类角色发出的。通常包含了用户想要模型回答或完成的请求。可以是一个问题、一段话、或者任何其他用户希望模型响应的内容
     - assistant：助手消息是模型的恢复
- content：消息就是传入模型的提示。

chat模型生成后的内容后，返回的响应，会包含一个或多个choices,包括以下字段：

- id：和上述的Text模型返回的结果含义一致
- object：和上述的Text模型返回的结果含义一致
- created：和上述的Text模型返回的结果含义一致
- model：和上述的Text模型返回的结果含义一致
- choices：返回的条目有所改变，包括：
  - message,message又包括：
    - role：消息的角色
    - content：消息的内容
  - finish_reason: 模型停止生成的原因，可能的值包括stop（遇到了停止标记）、length（达到了最大长度）或content_filter（被OpenAI内容过滤器移除）
  - index：表示这个选项在choices列表中的索引位置
- usage：和上述的Text模型返回的结果含义一致

### Text模型与Chat模型

各有优劣，Chat模型更适合处理多轮对话或多次交换的情况。Text模型更直接、更简单。

## 使用LangChain调用Text模型和Chat模型

