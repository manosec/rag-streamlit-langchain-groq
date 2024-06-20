from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader


def get_embedding_model(model="intfloat/e5-base-v2"):
    encoder_model = HuggingFaceEmbeddings(model=model)
    return encoder_model

class RAG:
    def __init__(self, chunk_size=512, chunk_overlap=10):
        self.splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.encoder_model = get_embedding_model()
        
    def chunk(document=''):
        self.loader = UnstructuredFileLoader(document)
        documents = self.loader.load()
        self.docs = self.splitter.split_documents(documents)
    
    def store():
        self.vector_store = FAISS.from_documents(self.docs, self.encoder_model)
        self.retriever = self.vector_store.as_retriever()
        
        
    