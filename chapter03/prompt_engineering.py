import json  # 用于解析JSON输出

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv(dotenv_path="../.env")

# 定义模板
template = """
你是一位专业的鲜花店文案撰写员。
对于售价{price}元的{flower_name},你能提供一个吸引人的简单描述吗？

{format_instructions}
"""

# 创建模型实例
model = ChatDeepSeek(model="deepseek-chat")

# 定义输出模式
output_parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="description", description="鲜花的描述文案"),
    ResponseSchema(name="reason", description="为什么要写这个文案")
])
format_instructions = output_parser.get_format_instructions()

# 创建提示模板 - 使用正确的方式添加部分变量
prompt = PromptTemplate(
    template=template,
    input_variables=["flower_name", "price"],
    partial_variables={"format_instructions": format_instructions}
)

print("提示模板已创建")

# 准备数据
flowers = ["玫瑰", "百合", "康乃馨"]
prices = [50, 30, 20]

# 创建DataFrame
df = pd.DataFrame(columns=["flower", "price", "description", "reason"])

# 处理每种鲜花
for flower, price in zip(flowers, prices):
    print(f"处理: {flower} - {price}元")

    # 格式化提示
    formatted_prompt = prompt.format(flower_name=flower, price=price)

    # 调用模型 - 获取AIMessage对象
    output = model.invoke(formatted_prompt)

    # 提取内容文本
    response_text = output.content

    # 调试输出
    print(f"原始响应: {response_text}")

    try:
        # 尝试解析结构化输出
        parsed_output = output_parser.parse(response_text)
    except Exception as e:
        print(f"解析失败: {str(e)}")
        # 尝试手动提取JSON
        try:
            # 查找JSON部分
            start_index = response_text.find('{')
            end_index = response_text.rfind('}') + 1
            json_str = response_text[start_index:end_index]
            parsed_output = json.loads(json_str)
        except Exception as json_e:
            print(f"JSON解析失败: {str(json_e)}")
            parsed_output = {
                "description": "生成失败",
                "reason": "解析错误"
            }

    # 添加到DataFrame
    row = {
        "flower": flower,
        "price": price,
        "description": parsed_output.get("description", ""),
        "reason": parsed_output.get("reason", "")
    }
    df.loc[len(df)] = row
    print(f"完成: {flower}")

# 输出结果
print("\n最终结果:")
print(df.to_dict(orient="records"))

# 保存到CSV
df.to_csv("flowers_with_descriptions.csv", index=False)
print("结果已保存到 flowers_with_descriptions.csv")
