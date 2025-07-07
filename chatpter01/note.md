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

## OpenAI API

LangChain本质上就是对各种大模型的API的套壳，是为了方便我们使用这些API，搭建起来的一些框架、模块和接口。

### 调用模型

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