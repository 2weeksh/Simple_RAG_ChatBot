from chatbot import RAGChatbot

if __name__ == "__main__":
    chatbot_instance = RAGChatbot(
        knowledge_file_path="2025 10 (공유) 강원대 중간 고사 내용.pdf"
    )
    chatbot_instance.start_chat()
