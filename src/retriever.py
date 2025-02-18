import os
import json
from typing import List, Optional

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        persist_path: str = "./data/vector_store",
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
            show_progress=True,
        )
        self.documents = self._get_documents()
        if not os.path.exists(persist_path):
            os.makedirs(os.path.dirname(persist_path), exist_ok=True)
            self.vector_store = SKLearnVectorStore.from_documents(self.documents, self.embeddings, persist_path=persist_path)
            self.vector_store.persist()
        else:
            self.vector_store = SKLearnVectorStore(persist_path=persist_path, embedding=self.embeddings)
        self.k = k
    
    def _conversation_to_documents(self, conversation: dict) -> List[Document]:
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
            doc
            for conversation in self.conversations
            for doc in self._conversation_to_documents(conversation)
        ]
    
    def similarity_search(self, query: str, k: Optional[int] = None, **kwargs) -> list[str]:
        if k is None:
            k = self.k
        return [
            doc.page_content
            for doc, _ in self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
        ]
    
    def messages_similarity_search(
        self, 
        messages: list[BaseMessage], 
        k: Optional[int] = None, 
        **kwargs
    ) -> list[str]:
        # Only keep HumanMessage and AIMessage
        messages = [message for message in messages if isinstance(message, (HumanMessage, AIMessage))]
        # Convert messages to strings
        messages_strings = []
        for message in messages:
            if isinstance(message, HumanMessage):
                messages_strings.append(f"HUMAN: {message.content}")
            else:
                messages_strings.append(f"GPT: {message.content}")
        return self.similarity_search(
            "\n".join(messages_strings),
            k=k,
            **kwargs
        )


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
