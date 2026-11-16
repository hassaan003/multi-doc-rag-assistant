from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
reply = llm.invoke("Explain RAG in 2 simple sentences.")
print(reply.content)