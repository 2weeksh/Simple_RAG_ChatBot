from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# db_manager.py 파일에서 함수를 import 합니다.
from db_manager import load_or_create_vector_db


class RAGChatbot:
    def __init__(self, knowledge_file_path, db_persist_directory="my_vector_db"):
        load_dotenv()

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", streaming=True)

        # DB
        self.vectorstore = load_or_create_vector_db(
            knowledge_file_path, db_persist_directory
        )

        self.retriever = self.vectorstore.as_retriever()
        self.prompt = self._create_prompt_template()
        self.rag_chain = self._create_rag_chain()
        self.chat_history = []
        print("챗봇입니다~.")

    def _create_prompt_template(self):
        system_prompt = """
        당신은 주어진 내용을 바탕으로 사용자의 질문에 답변하는 친절한 AI 어시스턴트입니다.
        이전 대화 내용을 참고하여 질문에 대한 답변을 생성하세요.
        주어진 내용에서만 답변을 찾아야 하며, 내용을 지어내서는 안 됩니다.

        내용:
        {context}
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    def _create_rag_chain(self):
        return (
            RunnablePassthrough.assign(
                context=lambda x: self.retriever.invoke(x["question"])
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, query):
        response_stream = self.rag_chain.stream(
            {"question": query, "chat_history": self.chat_history}
        )

        full_answer = ""
        print("답변: ", end="", flush=True)
        for chunk in response_stream:
            print(chunk, end="", flush=True)
            full_answer += chunk

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=full_answer))

    def start_chat(self):
        print("\n" + "=" * 30)
        print("🗣️ 챗봇과의 대화를 시작합니다.")
        print("챗봇을 종료하려면 'exit'을 입력하세요.")
        print("=" * 30)

        while True:
            query = input("질문: ")
            if query.lower() == "exit":
                print("챗봇을 종료합니다.")
                break
            self.ask(query)
            print("\n" + "-" * 50)
