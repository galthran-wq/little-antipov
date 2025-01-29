import logging
import json
import os
from typing import Annotated, Sequence, Literal, Optional, List
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.messages import ToolMessage, SystemMessage, BaseMessage, AIMessage, HumanMessage, trim_messages
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.config import load_config
from .redis_chain_memory import AsyncRedisSaver, RedisSaver

config = load_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if config.debug else logging.INFO)


class Chain:
    MAX_LENGTH = 110_000

    def __init__(self, model: str, system_prompt: str) -> None:
        self.model = model
        self.llm = OllamaLLM(
            model=model, 
        )
        self.system_prompt = system_prompt
    
    def _build_docs_chain(self, state: "State"):
        messages: List[BaseMessage] = state["messages"]
        system_prompt = self.get_system_message(state)
        prompt = ChatPromptTemplate.from_messages(
            [system_prompt] + messages[:-1] + [("user", messages[-1].content[0]["text"] + ":\n\n{context}")]
        )
        return create_stuff_documents_chain(llm=self.llm, prompt=prompt)

    class State(TypedDict):
        """LangGraph state."""
        messages: Annotated[Sequence[BaseMessage], add_messages]
        documents: Optional[Sequence[Document]] = None
        system: Optional[str] = None

    def get_system_message(self, state: "State"):
        # this is similar to customizing the create_react_agent with state_modifier, but is a lot more flexible
        system_prompt = self.system_prompt
        if state.get("system", None):
            system_prompt += f"\n\n{state['system']}"
        system_prompt = SystemMessage(system_prompt)
        return system_prompt

    def where_to_start(self, state: "State"):
        messages = state["messages"]
        last_message = messages[-1]
        assert isinstance(last_message, HumanMessage)
        if state.get("documents", None) is not None:
            return "docs_node"
        else:
            return "agent"

    def should_continue(self, state: "State") -> Literal["tools", "__end__"]:
        return "__end__"

    def trim_history(self, messages: list):
        return trim_messages(
            messages,
            # Keep the last <= n_count tokens of the messages.
            strategy="last",
            # Remember to adjust based on your model
            # or else pass a custom token_encoder
            token_counter=len,
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # Remember to adjust based on the desired conversation
            # length
            max_tokens=1,
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            start_on="human",
            # Most chat models expect that chat history ends with either:
            # (1) a HumanMessage or
            # (2) a ToolMessage
            end_on=("human", "tool"),
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
            allow_partial=False,
        )

    async def acall_model(self, state: "State", config: RunnableConfig):
        #  TODO: How does it work if graph.astream is used?? It is supposed to wait for the end of the full generation!
        # Somehow it knows that it shouldnt'
        system_prompt = self.get_system_message(state)
        messages = [system_prompt] + state["messages"]
        success = False
        while not success:
            try:
                response = await self.llm.ainvoke(messages, config)
                success = True
            except Exception as e:
                logger.debug(f"Error: {e}. Assuming it's context length exceeded")
                logger.debug(f"Trimming a message")
                messages = self.trim_history(messages)

        return {"messages": [response], "system": state.get("system", None)}
    
    async def docs(self, state: "State", config: RunnableConfig):
        chain = self._build_docs_chain(state["messages"])
        response = chain.invoke({"context": state["documents"]})
        return {"messages": [response], "system": state.get("system", None)}
    
    def build(self):
        # Initialize memory to persist state between graph runs
        if not config.debug:
            from redis.asyncio import Redis as AsyncRedis
            checkpointer = AsyncRedisSaver(
                conn=AsyncRedis(
                    host=config.redis_url, 
                    port=config.redis_port, 
                    db=config.redis_db,
                ),
            )
        else:
            checkpointer = MemorySaver()

        workflow = StateGraph(self.State)
        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.acall_model)
        workflow.add_node("docs_node", self.docs)

        workflow.add_conditional_edges("__start__", self.where_to_start)
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
        )
        workflow.add_edge("docs_node", "__end__")
        chain = workflow.compile(checkpointer=checkpointer)
        return chain
