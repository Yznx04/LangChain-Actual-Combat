from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path="../.env")
base_dir = Path(__file__).parent.joinpath("OneFlower")
documents = []

for file in base_dir.glob("**/*"):
    if file.suffix == ".pdf":
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    elif file.suffix == ".docx":
        loader = Docx2txtLoader(file)
        documents.extend(loader.load())
    elif file.suffix == ".txt":
        loader = TextLoader(file)
        documents.extend(loader.load())

# 将文档进行切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

vectorstore = Qdrant.from_documents(
    documents=chunked_documents,
    embedding=OpenAIEmbeddings(),  # 使用OpenAI的Embeddings Model做嵌入
    location=":memory:",  # 在内存中存储
    collection_name="my-documents",
)

# 相关信息获取

import logging
from langchain_deepseek import ChatDeepSeek
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

logging.basicConfig()
logging.getLogger("langchain.retrieval.multi_query").setLevel(logging.INFO)

# 实例化大模型工具

llm = ChatDeepSeek(model="deepseek-chat", temperature=0, max_tokens=2048)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_from_llm)

# 生成交互接口
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.post("/")
async def qa(question: str):
    result = qa_chain.invoke({"question": question})
    return result


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
