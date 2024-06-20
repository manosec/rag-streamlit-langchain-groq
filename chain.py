from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from rag import RAG
from llm import LLM


class Chain():
    def __init__(self, model:LLM, rag:RAG, prompt=(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )):
        
                
        self.model = model.llm
        self.rag = rag.langchain_retriever
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("human", "{input}"),
            ]
        )
        
        self.llm_chain = create_stuff_documents_chain(model.llm, self.prompt_template)
        self.rag_chain = create_retrieval_chain(self.rag, question_answer_chain)
        
    def query(self, query):
        response = self.rag_chain.invoke({"input": query})
        return response