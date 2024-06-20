from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()


class LLM:
    def __init__(self, groq_model='mixtral-8x7b-32768'):
        self.llm = ChatGroq(model=groq_model)
