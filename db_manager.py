import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_or_create_vector_db(data_path, persist_directory="my_vector_db"):
    """
    지정된 경로에 Vector DB가 없으면 새로 생성하고, 있으면 불러옵니다.
    """
    embeddings = OpenAIEmbeddings()

    if not os.path.exists(persist_directory):
        # DB 새로 생성

        # 문서 Load
        loader = PyPDFLoader(data_path)
        documents = loader.load()

        # 문서 Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_documents = text_splitter.split_documents(documents)

        # 벡터 저장
        vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        print("DB 생성이 완료되었습니다.")

    else:
        print(f"DB를 불러옵니다. 경로: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )

    return vectorstore
