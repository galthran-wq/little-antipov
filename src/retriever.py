import json
from typing import List, Optional
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import load_config

class Retriever:
    """Proxy class for SKLearnVectorStore"""
    def __init__(
        self, 
        conversations_paths: list[str],
        embedding_model: str = "BAAI/bge-m3",
        device: str = "cpu",
        k: int = 5,
    ):
        self.conversations_paths = conversations_paths
        self.conversations = [
            json.loads(line) 
            for path in conversations_paths 
            for line in open(path).readlines()
        ]
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
            show_progress_bar=True,
        )
        self.documents = self._get_documents()
        self.vector_store = SKLearnVectorStore.from_documents(self.documents, self.embeddings)
        self.k = k
    
    def _conversation_to_document(self, conversation: dict) -> List[Document]:
        context_list = []
        current_context = []

        for message in conversation:
            current_context.append(f"{message['from'].upper()}: {message['value']}")
            if message['from'] == 'gpt' and len(current_context) > 1:
                context_list.append(Document(page_content='\n'.join(current_context)))
                current_context = []
        return context_list
    
    def _get_documents(self):
        return [
            self._conversation_to_document(conversation)
            for conversation in self.conversations
        ]
    
    def similarity_search(self, query: str, k: Optional[int] = None, **kwargs) -> list[str]:
        if k is None:
            k = self.k
        return [
            doc.page_content
            for doc, _ in self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
        ]



def setup_retriever() -> Optional[Retriever]:
    config = load_config()
    if config.retriever is None:
        return None
    return Retriever(
        conversations_paths=config.retriever.conversations_paths,
        embedding_model=config.retriever.embedding_model,
        device=config.retriever.device,
        k=config.retriever.k,
    )
