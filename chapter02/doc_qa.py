import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI  # 用于兼容 OpenAI 的 API 方式
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# 加载环境变量
load_dotenv(dotenv_path="../.env")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("DeepSeek API密钥未设置。请在.env文件中添加DEEPSEEK_API_KEY")

base_dir = Path(__file__).parent.joinpath("OneFlower")
documents = []

print(f"正在加载文档，目录: {base_dir}")

for file in base_dir.glob("**/*"):
    if file.suffix == ".pdf":
        print(f"加载PDF: {file.name}")
        loader = PyPDFLoader(str(file))
        documents.extend(loader.load())
    elif file.suffix == ".docx":
        print(f"加载DOCX: {file.name}")
        loader = Docx2txtLoader(str(file))
        documents.extend(loader.load())
    elif file.suffix == ".txt":
        print(f"加载TXT: {file.name}")
        loader = TextLoader(str(file))
        documents.extend(loader.load())

print(f"共加载 {len(documents)} 个文档")

# 将文档进行切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
)
chunked_documents = text_splitter.split_documents(documents)
print(f"文档切分为 {len(chunked_documents)} 个片段")

# 使用开源的Embeddings模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("正在创建向量数据库...")
vectorstore = Qdrant.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="my-documents",
)
print("向量数据库创建完成")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langchain")
logger.setLevel(logging.INFO)


# 解决方案2：使用兼容OpenAI API的方式调用DeepSeek
llm = ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",  # DeepSeek的API端点
    temperature=0.1,
    max_tokens=2048
)

# 创建基础检索器
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

# 创建自定义提示模板
template = """
基于以下上下文信息回答用户的问题。如果不知道答案，就回答不知道。

上下文:
{context}

问题: {question}

请用中文给出有帮助的回答:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=base_retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt
    }
)

app = FastAPI()


@app.post("/")
async def qa(question: str):
    print(f"收到问题: {question}")
    try:
        result = qa_chain.invoke({"query": question})
        print(f"问题处理完成: {question}")

        # 返回结构化响应
        return {
            "answer": result["result"],
            "sources": [doc.metadata["source"] for doc in result["source_documents"]]
        }
    except Exception as e:
        print(f"处理问题时出错: {str(e)}")
        return {"error": str(e)}


if __name__ == '__main__':
    print("启动服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
